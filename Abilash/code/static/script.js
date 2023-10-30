let isCallEnded = false;

// Function to end the call
function endCall() {
    console.log("End Call button clicked.");

    // Set the flag to indicate that the call has ended
    isCallEnded = true;

    // Hide the video element
    const video = document.getElementById('video_feed');
    video.style.display = "none";

    // Send a POST request to the Flask server to end the call
    fetch('/end_call', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        // When the response is received, display the message
        alert(data.message);
        // Update the stress percentage in the frontend
        document.getElementById("stressPercentage").innerText = `Your stress Percentage: ${data.stress_percentage.toFixed(2)}%`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Function to check if the call has ended
function checkCallEnded() {
    if (isCallEnded) {
        // Stop the video stream by setting the source to an empty string
        const video = document.getElementById('video_feed');
        video.src = '';
    } else {
        // Continue checking if the call has ended every 1 second
        setTimeout(checkCallEnded, 1000);
    }
}

// Start checking if the call has ended when the page is loaded
document.addEventListener('DOMContentLoaded', checkCallEnded);

// Function to download the CSV file
function downloadCsv() {
    // Create a link element to trigger the download
    const link = document.createElement('a');
    link.href = '/download_csv'; // Use the correct route URL
    link.download = 'student_data.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function downloadAttendanceReport() {
    // Create a link element to trigger the download
    const link = document.createElement('a');
    link.href = '/download_attendance_report'; // Use the correct route URL
    link.download = 'attendance_report.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

 // Attach click event listener to "Stress Detection" link
    document.getElementById('stressDetectionLink').addEventListener('click', function() {
        // Reload the page
        window.location.reload();
    });