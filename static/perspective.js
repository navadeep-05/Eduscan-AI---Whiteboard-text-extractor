
/*
EduScan manual perspective selection.

The user clicks four board corners in this order:
1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

Coordinates are converted from the preview canvas
back to the original image's pixel coordinates.
*/

document.addEventListener("DOMContentLoaded", () => {
    const image = document.getElementById("perspective-source");
    const canvas = document.getElementById("perspective-canvas");
    const resetButton = document.getElementById("reset-corners");
    const applyButton = document.getElementById("apply-perspective");
    const status = document.getElementById("perspective-status");
    const cornersInput = document.getElementById("corners");

    // The feature is not present on every page yet.
    if (!image || !canvas) return;

    const context = canvas.getContext("2d");
    const points = [];

    function drawCanvas() {
        const maxWidth = 800;
        const scale = Math.min(1, maxWidth / image.naturalWidth);

        canvas.width = Math.round(image.naturalWidth * scale);
        canvas.height = Math.round(image.naturalHeight * scale);

        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);

        // Draw the selected corner points.
        points.forEach((point, index) => {
            context.beginPath();
            context.arc(point.x, point.y, 7, 0, Math.PI * 2);
            context.fillStyle = "#ef4444";
            context.fill();

            context.fillStyle = "#ffffff";
            context.font = "bold 13px sans-serif";
            context.fillText(String(index + 1), point.x + 10, point.y - 10);
        });

        // Connect selected corners in the order they were clicked.
        if (points.length > 1) {
            context.beginPath();
            context.moveTo(points[0].x, points[0].y);

            points.slice(1).forEach(point => {
                context.lineTo(point.x, point.y);
            });

            // Close the outline after the fourth corner.
            if (points.length === 4) {
                context.closePath();
            }

            context.strokeStyle = "#22c55e";
            context.lineWidth = 3;
            context.stroke();
        }

        // Convert preview coordinates to original image coordinates.
        const scaleX = image.naturalWidth / canvas.width;
        const scaleY = image.naturalHeight / canvas.height;

        const originalCoordinates = points.map(point => ({
            x: Math.round(point.x * scaleX),
            y: Math.round(point.y * scaleY)
        }));

        cornersInput.value = JSON.stringify(originalCoordinates);

        applyButton.disabled = points.length !== 4;

        status.textContent = points.length < 4
            ? `Select corner ${points.length + 1} of 4.`
            : "Four corners selected. You can apply correction.";
    }

    canvas.addEventListener("click", event => {
        if (points.length >= 4) return;

        const rect = canvas.getBoundingClientRect();

        // Account for the canvas being displayed at a CSS-scaled size.
        const x = (event.clientX - rect.left) * canvas.width / rect.width;
        const y = (event.clientY - rect.top) * canvas.height / rect.height;

        points.push({ x, y });
        drawCanvas();
    });

    resetButton?.addEventListener("click", () => {
        points.length = 0;
        drawCanvas();
    });

    image.addEventListener("load", drawCanvas);

    if (image.complete && image.naturalWidth > 0) {
        drawCanvas();
    }
});