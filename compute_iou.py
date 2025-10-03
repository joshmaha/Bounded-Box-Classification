import numpy as np	

def match(img_shape, cx, cy, h, a, gt_cx, gt_cy, gt_h, gt_a):
	mask = np.zeros(img_shape[:2], dtype=np.uint8)

	sina = np.sin(np.pi*a/180.0)
	cosa = np.cos(np.pi*a/180.0)

	vx, vy = -sina*h/2.0, cosa*h/2.0
	hx, hy = cosa*0.6*h/2.0, sina*0.6*h/2.0

	pts = [[cx-vx-hx,cy-vy-hy],[cx-vx+hx,cy-vy+hy],[cx+vx+hx,cy+vy+hy],[cx+vx-hx,cy+vy-hy]]

	for y in range(0,img_shape[0],3):
		for x in range(0,img_shape[1],3):
			flag = True
			for i in range(4):
				p1 = pts[i]
				p2 = pts[(i+1)%4]
				if (p2[1]-p1[1])*x - (p2[0]-p1[0])*y + p2[0]*p1[1] - p2[1]*p1[0] > 0:
					flag = False
			if flag:
				mask[y,x] = 1

	mask2 = np.zeros(img_shape[:2], dtype=np.uint8)

	sina = np.sin(np.pi*gt_a/180.0)
	cosa = np.cos(np.pi*gt_a/180.0)

	vx, vy = -sina*gt_h/2.0, cosa*gt_h/2.0
	hx, hy = cosa*0.6*gt_h/2.0, sina*0.6*gt_h/2.0

	pts = [[gt_cx-vx-hx,gt_cy-vy-hy],[gt_cx-vx+hx,gt_cy-vy+hy],[gt_cx+vx+hx,gt_cy+vy+hy],[gt_cx+vx-hx,gt_cy+vy-hy]]

	for y in range(0,img_shape[0],3):
		for x in range(0,img_shape[1],3):
			flag = True
			for i in range(4):
				p1 = pts[i]
				p2 = pts[(i+1)%4]
				if (p2[1]-p1[1])*x - (p2[0]-p1[0])*y + p2[0]*p1[1] - p2[1]*p1[0] > 0:
					flag = False
			if flag:
				mask2[y,x] = 1

	iou = np.sum(mask*mask2)/np.sum(np.bitwise_or(mask,mask2))
	
	return iou

