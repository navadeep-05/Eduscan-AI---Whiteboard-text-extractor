import cv2
import numpy as np


def load_image(image_path):
    """Load an image and raise an error if the path is invalid."""

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def correct_perspective(image, corners):
    """
    Flatten a tilted whiteboard using four corner coordinates.

    corners order:
    top-left, top-right, bottom-right, bottom-left.
    """

    # Convert the selected coordinates into OpenCV's float format.
    source = np.float32(corners)

    # Identify the four selected corners.
    top_left, top_right, bottom_right, bottom_left = source

    # Estimate the width and height of the flattened board.
    width = int(max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left)
    ))

    height = int(max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right)
    ))

    # Prevent invalid output dimensions.
    if width < 2 or height < 2:
        raise ValueError("The selected corners are too close together.")

    # Define the four corners of a rectangular output image.
    destination = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    # Calculate the perspective mapping from source to destination.
    matrix = cv2.getPerspectiveTransform(source, destination)

    # Apply the mapping to flatten the board.
    corrected = cv2.warpPerspective(image, matrix, (width, height))

    return corrected


def enhance_image(image):
    """Create grayscale and contrast-enhanced versions."""

    # Convert the colour image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve local contrast so faint marker strokes may become clearer.
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # Return both versions so we can compare their OCR performance.
    return gray, enhanced


def create_threshold_image(enhanced):
    """Create an optional black-and-white version for OCR experiments."""

    # Use local brightness to classify pixels as black or white.
    # This may help some images, but can also amplify noise.
    threshold = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    return threshold

def order_board_corners(points):
    """
    Arrange four corner points in this order:
    top-left, top-right, bottom-right, bottom-left.

    Args:
        points: Four (x, y) corner coordinates.

    Returns:
        NumPy array containing the ordered corner coordinates.
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    # Smallest x + y is usually the top-left corner.
    # Largest x + y is usually the bottom-right corner.
    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    # Smallest x - y is usually top-right.
    # Largest x - y is usually bottom-left.
    differences = points[:, 0] - points[:, 1]
    ordered[1] = points[np.argmax(differences)]
    ordered[3] = points[np.argmin(differences)]

    return ordered


def detect_board_corners(image):
    """
    Try to detect a whiteboard using OpenCV contours.

    Args:
        image: Original image loaded as a NumPy array.

    Returns:
        Four ordered corner coordinates if detection succeeds.
        None if no suitable quadrilateral is found.

    Note:
        This is a best-effort detector. It may fail when the
        board has weak borders, glare, clutter, or is cropped.
    """
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    image_area = height * width

    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce small image noise before finding edges.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find strong edges in the image.
    edges = cv2.Canny(blurred, 50, 150)

    # Connect nearby edge segments to improve contour detection.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel
    )

    # Retrieve contours from the edge image.
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Test larger contours first.
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours[:30]:
        area = cv2.contourArea(contour)

        # Ignore contours that are too small to be a board.
        if area < image_area * 0.10:
            continue

        perimeter = cv2.arcLength(contour, True)

        # Approximate the contour with fewer points.
        polygon = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # A board candidate must have four convex corners.
        if len(polygon) != 4:
            continue

        if not cv2.isContourConvex(polygon):
            continue

        corners = polygon.reshape(4, 2)

        # Reject extremely narrow or degenerate candidates.
        rectangle = cv2.minAreaRect(corners)
        rect_width, rect_height = rectangle[1]

        if min(rect_width, rect_height) < 1:
            continue

        aspect_ratio = max(
            rect_width, rect_height
        ) / min(rect_width, rect_height)

        if aspect_ratio > 5:
            continue

        return order_board_corners(corners)

    # No suitable board-like contour was found.
    return None

# Run a small test when this file is executed directly.
if __name__ == "__main__":

    # Load the perspective-corrected image created in Exercise 3.
    image = load_image("assets/perspective_transformed_images/perspective_corrected.jpg")

    # Generate grayscale and CLAHE-enhanced versions.
    gray, enhanced = enhance_image(image)

    # Generate the optional thresholded version.
    threshold = create_threshold_image(enhanced)

    # Save the outputs for later comparison and OCR testing.
    cv2.imwrite("grayscale_whiteboard.jpg", gray)
    cv2.imwrite("enhanced_whiteboard.jpg", enhanced)
    cv2.imwrite("threshold_whiteboard.jpg", threshold)

    print("Image processing completed successfully!")
