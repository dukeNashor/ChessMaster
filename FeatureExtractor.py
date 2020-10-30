import cv2


g_sift_extractor = sift = cv2.SIFT_create(edgeThreshold = 0)


class FeatureExtractor:

    # reference:
    # SURF: https://docs.opencv.org/master/df/dd2/tutorial_py_surf_intro.html


    @staticmethod
    # OpenCV-python wrapper for a single SIFT calculation on a grid of a board image.
    # ref: https://docs.opencv.org/4.4.0/d7/d60/classcv_1_1SIFT.html
    # In our case, the direction is manually assigned (using default angle = 0), e.g. 12 o'clock direction.
    # This ensures the consistency among the same type of chess piece images.
    def ExtractSIFTForGrid(board_image, row, col, center_x = 25, center_y = 25, radius = 45):
        kps = [cv2.KeyPoint(x = center_x + 50 * col, y = center_y + 50 * row, _size = 45)]
        keypoints, descriptors = g_sift_extractor.compute(image = board_image, keypoints = kps)
        return keypoints[0], descriptors[0, :]




