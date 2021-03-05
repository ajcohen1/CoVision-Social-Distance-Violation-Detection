from collections import Counter

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
#from deep_sort.utils.parser import get_config
#from deep_sort.deep_sort import DeepSort
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import argparse
import os
import platform
import shutil
from birdeye_utils import birdeye_transformer
from birdeye_utils import birdeye_video_writer
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
# https://github.com/pytorch/pytorch/issues/3678
import sys
import numpy as np
from aux_functions import *
from dynamic_plotter import *

sys.path.insert(0, './yolov5')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
mouse_pts = []

SHRINK_X = 0.1
SHRINK_Y = 0.1

image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)

window_size = 5
person_list = {}

def get_coords_avg(avg_list):
    avg_x = 0
    avg_y = 0
    for past_coords in avg_list:
        if past_coords[0] == -1 and past_coords[1] == -1:
            continue
        avg_x += past_coords[0]
        avg_y += past_coords[1]
    avg_x /= window_size
    avg_y /= window_size

    return avg_x, avg_y

def resize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

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

    for index, box in enumerate(xywh):
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


def compute_color_for_labels(risk_factor):
    """
    Simple function that adds fixed color depending on the class
    """
    if (risk_factor) > 1:
        return [0, 0, 255]
    return [0, 128, 0]


def draw_boxes(img, bbox, cluster_labels=None, offset=(0, 0)):
    j = len(cluster_labels)
    # key: id, val: [x, y, cluster id, tsize]
    xy_coords = {}
    # key: user id; value: cluster id
    cluster_dict = {}
    # key: cluster id; value: score
    cluster_score = {}
    for i, box in enumerate(bbox):
        # if i == j:
        #    break
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        cluster_id = int(cluster_labels[i]) * 7 if cluster_labels is not None else 0

        cluster_dict[i] = cluster_id
        # print(cluster_dict)

        cluster_score = Counter(cluster_dict.values())
        # print(cluster_score)
        """if (cluster_id in cluster_dict):
            if not(id in cluster_dict[cluster_id]):
                cluster_dict[cluster_id].append(id)
        else:
            cluster_dict[cluster_id] = [id]
        print("Cluster Dict: ")
        print(cluster_dict)"""
        color = compute_color_for_labels(cluster_id)

        label = '{}{:d}'.format("", i)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        xy_coords[i] = [x1, y1, x2, y2, cluster_id, t_size[1]]
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        # cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    for cID in cluster_dict:
        risk = 0
        newx1 = xy_coords[cID][0]
        newy1 = xy_coords[cID][1]
        newx2 = xy_coords[cID][2]
        newy2 = xy_coords[cID][3]
        if xy_coords[cID][4] in cluster_score:
            risk = cluster_score[xy_coords[cID][4]]
        color = compute_color_for_labels(risk)
        new_label = '{}{:d}'.format("", risk)
        new_label = "Risk: " + new_label
        cv2.rectangle(img, (newx1, newy1), (newx2, newy2), color, 3)
        cv2.putText(img, new_label, (newx1, newy1 + xy_coords[cID][5] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255],
                    2)

    return img

def remove_points_outside_ROI(outputs, ROI_polygon):
    points_inside_ROI = []
    for point in enumerate(outputs):
        if point_within_ROI(point[1], ROI_polygon):
            points_inside_ROI.append(list(point[1][:4]))
    return points_inside_ROI

def compute_frame_rf(risk_dict):
    return sum(risk_dict.values()) + len(risk_dict.values())

def detect(opt, save_img=False):
    global bird_image
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize the ROI frame
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)

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

    d = DynamicUpdate()
    d.on_launch()

    risk_factors = []
    frame_nums = []

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

        # initialize birdeye view video writer
        frame_h, frame_w, _ = image.shape
        bevw = birdeye_video_writer.birdeye_video_writer(frame_h, frame_w, M, threshold_pixel_dist, ROI_pts)

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

                bbox_xywh = []
                bbox_xyxy = []

                ROI_polygon = Polygon(ROI_pts)

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]

                    bbox_xyxy.append(xyxy)
                    bbox_xywh.append(obj)

                # draw boxes for visualization
                if len(bbox_xywh) > 0:
                    # filter deepsort output
                    outputs_in_ROI = remove_points_outside_ROI(bbox_xyxy, ROI_polygon)
                    tlbr_in_ROI = outputs_in_ROI
                    center_coords_in_ROI = xywh_to_center_coords(outputs_in_ROI)
                    print("Center Coords: ", center_coords_in_ROI)

                    warped_pts = birdeye_transformer.transform_center_coords_to_birdeye(center_coords_in_ROI, M)

                    clusters = DBSCAN(eps=threshold_pixel_dist, min_samples=1).fit(warped_pts)
                    draw_boxes(im0, outputs_in_ROI, clusters.labels_)

                    # embded the bird image to the video
                    risk_dict = Counter(clusters.labels_)
                    bird_image = bevw.create_birdeye_frame(warped_pts, clusters.labels_, risk_dict)
                    bird_image = resize(bird_image, 30)
                    bv_height, bv_width, _ = bird_image.shape
                    frame_x_center, frame_y_center = frame_w //2, frame_h//2
                    x_offset = 20


                    im0[ frame_y_center-bv_height//2:frame_y_center+bv_height//2, \
                        x_offset:bv_width+x_offset ] = bird_image


                    #write the risk graph

                    risk_factors += [compute_frame_rf(risk_dict)]
                    frame_nums += [frame_idx]
                    d.on_running(frame_nums, risk_factors)

                # Write MOT compliant results to file
                if save_txt and len(outputs_in_ROI) != 0:
                    for j, output in enumerate(outputs_in_ROI):
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
                    cv2.imwrite(save_path, bird_image)
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
