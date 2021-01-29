from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import argparse
import os
import platform
import shutil
import matplotlib.pyplot as plt
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
# https://github.com/pytorch/pytorch/issues/3678
import sys
import numpy as np
from aux_functions import *

sys.path.insert(0, './yolov5')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
mouse_pts = []

SHRINK_X = 0.1
SHRINK_Y = 0.1

image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)

plt.ion()

class DynamicUpdate():
    # Suppose we know the x range
    min_x = 0
    max_x = 2000
    min_y = 0
    max_y = 2000

    def on_launch(self):
        # Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], 'o')
        # Autoscale on unknown axis and known lims on the other
        #self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        # Other stuff
        self.ax.grid()
        ...

    def on_running(self, xdata, ydata):
        # Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update_pts(self, warped_pts, clusters):
        xdata = []
        ydata = []
        for i, pt in enumerate(warped_pts):
            xdata.append(pt[0])
            ydata.append(pt[1])
        print("xdata: " + repr(xdata))
        print("ydata: " + repr(ydata))
        self.on_running(xdata, ydata)

d = DynamicUpdate()
d.on_launch()

def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))


def xywh_to_center_coords(xywh):
    xyxy = []

    for box_ in enumerate(xywh):
        box = box_[1]
        tl_coord = (box[0], box[1])
        br_coord = (box[2], box[3])
        x_center = int((tl_coord[0] + br_coord[0]) / 2)
        y_center = int((tl_coord[1] + br_coord[1]) / 2)
        center_coords = (x_center, y_center)
        xyxy.append(center_coords)
    return xyxy


def point_within_ROI(deepsort_output_pt, ROI_polygon):
    tl_coord = (deepsort_output_pt[0], deepsort_output_pt[1])
    br_coord = (deepsort_output_pt[2], deepsort_output_pt[3])
    x_center = (int)((tl_coord[0] + br_coord[0]) / 2)
    y_center = (int)((tl_coord[1] + br_coord[1]) / 2)
    center_coord = Point(x_center, y_center)
    return ROI_polygon.contains(center_coord)


def shrink_ROI_pts(coords, x_shrink, y_shrink):
    xs = [i[0] for i in coords]
    ys = [i[1] for i in coords]

    # simplistic way of calculating a center of the graph, you can choose your own system
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)

    # shrink figure
    new_xs = [(i - x_center) * (1 - x_shrink) + x_center for i in xs]
    new_ys = [(i - y_center) * (1 - y_shrink) + y_center for i in ys]

    # create list of new coordinates
    new_coords = zip(new_xs, new_ys)
    return list(new_coords)


def bbox_rel(image_width, image_height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, cluster_labels=None, offset=(0, 0)):
    j = len(cluster_labels)
    for i, box in enumerate(bbox):
        # if i == j:
        #    break
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        cluster_id = int(cluster_labels[i]) * 7 if cluster_labels is not None else 0
        color = compute_color_for_labels(cluster_id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img


def remove_points_outside_ROI(outputs, ROI_polygon):
    points_inside_ROI = []
    ids_inside_ROI = []
    for point in enumerate(outputs):
        if point_within_ROI(point[1], ROI_polygon):
            points_inside_ROI.append(list(point[1][:4]))
            ids_inside_ROI.append(point[1][-1])

    return (points_inside_ROI, ids_inside_ROI)


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize the ROI frame
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Get the ROI if this is the first frame
        if frame_idx == 0:
            while True:
                image = im0s
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 7:
                    cv2.destroyWindow("image")
                    break
                first_frame_display = False
        four_points = mouse_pts

        # Get perspective, M is the transformation matrix for bird's eye view
        M, Minv = get_camera_perspective(image, four_points[0:4])

        # Last two points in getMousePoints... this will be the threshold distance between points
        threshold_pts = src = np.float32(np.array([four_points[4:]]))

        # Convert distance to bird's eye view
        warped_threshold_pts = cv2.perspectiveTransform(threshold_pts, M)[0]

        # Get distance in pixels
        threshold_pixel_dist = np.sqrt(
            (warped_threshold_pts[0][0] - warped_threshold_pts[1][0]) ** 2
            + (warped_threshold_pts[0][1] - warped_threshold_pts[1][1]) ** 2
        )

        # Draw the ROI on the output images
        ROI_pts = np.array(
            [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
        )
        cv2.polylines(im0s, [ROI_pts], True, (0, 255, 255), thickness=4)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                person_center_coords = []
                confs = []

                ROI_polygon = Polygon(ROI_pts)

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)
                # print("Output len: ", outputs[:, -1])
                # draw boxes for visualization
                if len(outputs) > 0:

                    # filter deepsort output
                    outputs_in_ROI = remove_points_outside_ROI(outputs, ROI_polygon)
                    xywh_in_ROI = outputs_in_ROI[0]
                    ids_in_ROI = outputs_in_ROI[1]
                    center_coords_in_ROI = xywh_to_center_coords(xywh_in_ROI)

                    # convert all center coordinates to birds view
                    warped_pts = plot_points_on_bird_eye_view(
                        im0, center_coords_in_ROI, M, 1, 1
                    )

                    color = (0, 255, 0)
                    #bird_image = cv2.line(bird_image,
                    #                      (warped_threshold_pts[0][0], warped_threshold_pts[0][1]),
                    #                      (warped_threshold_pts[1][0], warped_threshold_pts[1][1]),
                    #                      color,
                    #                      5)

                    #a_ = plot_lines_between_nodes(warped_pts, bird_image, threshold_pixel_dist)

                    # time for the dbscan to get the cluster groups
                    # clusters = AgglomerativeClustering(None, 'euclidean', None, None, 'auto', "single", threshold_pixel_dist).fit(warped_pts)
                    clusters = DBSCAN(eps=threshold_pixel_dist, min_samples=1).fit(warped_pts)

                    bbox_xyxy = xywh_in_ROI
                    identities = ids_in_ROI
                    draw_boxes(im0, bbox_xyxy, identities, clusters.labels_)

                    d.update_pts(warped_pts, clusters)
                    print("HERE--------------------------")

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                # cv2.imshow("bird_image", bird_image)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5x.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
