// Statistics tracking
let stats = {
    frameCount: 0,
    fps: 0,
    maxFps: 0,
    totalProcessingTime: 0,
    avgProcessingTime: 0,
    lastUpdate: Date.now()
};

let updateInterval = 500; // milliseconds

function updateStats() {
    fetch('/stats')
        .then(response => response.json())
        .then(data => {
            // Update display values
            document.getElementById('fps').textContent = data.fps;
            document.getElementById('landmarks').textContent = data.landmarks_detected;
            document.getElementById('emotion').textContent = data.emotion || '--';
            document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
            
            // Update emotion panel
            if (data.emotion) {
                const emotionText = document.getElementById('emotionText');
                emotionText.textContent = data.emotion;
                
                // Add stability indicator
                if (data.emotion_stable) {
                    emotionText.innerHTML = data.emotion + ' <span style="color: #4CAF50; font-size: 0.8em;">✓</span>';
                } else {
                    emotionText.innerHTML = data.emotion + ' <span style="color: #FFC107; font-size: 0.8em;">~</span>';
                }
                
                const confidencePercent = Math.min(100, data.confidence * 100);
                document.getElementById('confidenceFill').style.width = confidencePercent + '%';
                document.getElementById('confidenceText').textContent = confidencePercent.toFixed(1) + '% confidence';
            } else {
                document.getElementById('emotionText').textContent = '--';
                document.getElementById('confidenceFill').style.width = '0%';
                document.getElementById('confidenceText').textContent = '0% confidence';
            }

            // Update emotion ranking list
            const rankList = document.getElementById('emotionRankList');
            if (data.emotion_ranking && data.emotion_ranking.length > 0) {
                rankList.innerHTML = '';
                data.emotion_ranking.slice(0, 5).forEach((entry) => {
                    const emotion = entry[0];
                    const score = Number(entry[1] || 0);
                    const item = document.createElement('li');
                    item.textContent = `${emotion}: ${(score * 100).toFixed(1)}%`;
                    rankList.appendChild(item);
                });
            } else {
                rankList.innerHTML = '<li>No ranking available</li>';
            }

            // Update status
            const cameraStatus = document.getElementById('cameraStatus');
            const detectionStatus = document.getElementById('detectionStatus');

            if (data.camera_active) {
                cameraStatus.innerHTML = '🟢 Connected';
                cameraStatus.className = 'status-value status-active';
            } else {
                cameraStatus.innerHTML = '🔴 Disconnected';
                cameraStatus.className = 'status-value status-error';
            }

            if (data.landmarks_detected > 0) {
                detectionStatus.innerHTML = '✅ Face Detected';
                detectionStatus.className = 'status-value status-active';
            } else {
                detectionStatus.innerHTML = '⭕ No Face';
                detectionStatus.className = 'status-value status-inactive';
            }

            // Update statistics
            stats.fps = data.fps;
            stats.frameCount += 1;
            
            if (stats.fps > stats.maxFps) {
                stats.maxFps = stats.fps;
            }

            stats.totalProcessingTime += data.processing_time_ms;
            stats.avgProcessingTime = stats.totalProcessingTime / stats.frameCount;

            // Update stat cards
            const avgFps = Math.round((stats.fps + stats.fps) / 2); // Simplified average
            document.getElementById('avgFps').textContent = stats.fps;
            document.getElementById('totalFrames').textContent = stats.frameCount;
            document.getElementById('avgProcessing').textContent = stats.avgProcessingTime.toFixed(2) + 'ms';
            document.getElementById('peakFps').textContent = stats.maxFps;
        })
        .catch(error => {
            console.error('Error fetching stats:', error);
            document.getElementById('cameraStatus').innerHTML = '❌ Error';
            document.getElementById('cameraStatus').className = 'status-value status-error';
        });
}

function checkHealth() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'healthy') {
                console.warn('Server health check failed');
            }
        })
        .catch(error => {
            console.error('Health check error:', error);
        });
}

// Event Listeners
document.getElementById('refreshBtn').addEventListener('click', () => {
    updateStats();
    console.log('Stats refreshed manually');
});

document.getElementById('updateInterval').addEventListener('change', (e) => {
    updateInterval = parseInt(e.target.value);
    console.log(`Update interval changed to ${updateInterval}ms`);
    
    // Restart the update loop
    clearInterval(window.updateLoop);
    window.updateLoop = setInterval(updateStats, updateInterval);
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Expression Detector loaded');

    // Initial update
    updateStats();
    checkHealth();

    // Set up periodic updates
    window.updateLoop = setInterval(updateStats, updateInterval);

    // Health check every 30 seconds
    setInterval(checkHealth, 30000);

    // Log when video stream connects
    const videoFeed = document.getElementById('videoFeed');
    videoFeed.addEventListener('load', () => {
        console.log('Video feed connected');
    });

    videoFeed.addEventListener('error', () => {
        console.error('Video feed error');
        document.getElementById('cameraStatus').innerHTML = '❌ Stream Error';
        document.getElementById('cameraStatus').className = 'status-value status-error';
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.updateLoop) {
        clearInterval(window.updateLoop);
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // 'R' key to refresh stats
    if (e.key === 'r' || e.key === 'R') {
        updateStats();
    }
    // 'F' key to toggle fullscreen (would need additional implementation)
    if (e.key === 'f' || e.key === 'F') {
        const videoContainer = document.querySelector('.video-container');
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            videoContainer.requestFullscreen().catch(err => {
                console.warn('Fullscreen request failed:', err);
            });
        }
    }
});
