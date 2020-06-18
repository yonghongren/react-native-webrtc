package com.oney.WebRTCModule;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.WindowManager;

import com.facebook.react.bridge.ActivityEventListener;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.BaseActivityEventListener;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.WritableMap;

import org.webrtc.MediaStream;
import org.webrtc.PeerConnectionFactory;
import org.webrtc.ScreenCapturerAndroid;
import org.webrtc.VideoCapturer;
import org.webrtc.VideoSource;
import org.webrtc.VideoTrack;
import org.webrtc.*;

import java.util.UUID;

/**
 * The implementation of {@code getDisplayMedia} extracted into a separate file in
 * order to reduce complexity and to (somewhat) separate concerns.
 */
class GetDisplayMediaImpl {
    /**
     * The request code identifying requests for the permission to capture
     * the screen. The value must be 16-bit and is arbitrarily chosen here.
     */
    private static final int PERMISSION_REQUEST_CODE = (int) (Math.random() * Short.MAX_VALUE);

    /**
     * The {@link Log} tag with which {@code GetDisplayMediaImpl} is to log.
     */
    private static final String TAG = WebRTCModule.TAG;

    private final ReactApplicationContext reactContext;
    private final WebRTCModule webRTCModule;
    private final ActivityEventListener activityEventListener;

    private Promise promise;
    private Intent mediaProjectionPermissionResultData;

    GetDisplayMediaImpl(WebRTCModule webRTCModule, ReactApplicationContext reactContext) {
        this.webRTCModule = webRTCModule;
        this.reactContext = reactContext;

        this.activityEventListener = new BaseActivityEventListener() {
            @Override
            public void onActivityResult(Activity activity, int requestCode, int resultCode, Intent data) {
                super.onActivityResult(activity, requestCode, resultCode, data);
                if (requestCode == PERMISSION_REQUEST_CODE) {
                    if (resultCode != Activity.RESULT_OK) {
                        promise.reject("DOMException", "NotAllowedError");
                        promise = null;
                        return;
                    }

                    mediaProjectionPermissionResultData = data;
                    createStream();
                }
            }
        };

        reactContext.addActivityEventListener(this.activityEventListener);
    }

    public void getDisplayMedia(Promise promise) {
        if (this.promise != null) {
            promise.reject(new RuntimeException("Another operation is pending."));
            return;
        }

        Activity currentActivity = this.reactContext.getCurrentActivity();
        if (currentActivity == null) {
            promise.reject(new RuntimeException("No current Activity."));
            return;
        }

        this.promise = promise;

        MediaProjectionManager mediaProjectionManager =
            (MediaProjectionManager) currentActivity.getApplication().getSystemService(
                Context.MEDIA_PROJECTION_SERVICE);
        currentActivity.startActivityForResult(
            mediaProjectionManager.createScreenCaptureIntent(), PERMISSION_REQUEST_CODE);
    }

    private void createStream() {
        String streamId = UUID.randomUUID().toString();
        MediaStream mediaStream = webRTCModule.mFactory.createLocalMediaStream(streamId);
        VideoTrack track = createTrack();
        mediaStream.addTrack(track);

        WritableMap data = Arguments.createMap();
        data.putString("streamId", streamId);

        WritableMap trackInfo = Arguments.createMap();
        String trackId = track.id();

        trackInfo.putBoolean("enabled", track.enabled());
        trackInfo.putString("id", trackId);
        trackInfo.putString("kind", track.kind());
        trackInfo.putString("label", trackId);
        trackInfo.putString("readyState", track.state().toString());
        trackInfo.putBoolean("remote", false);

        data.putMap("track", trackInfo);

        webRTCModule.localStreams.put(streamId, mediaStream);
        promise.resolve(data);

        // Cleanup
        mediaProjectionPermissionResultData = null;
        promise = null;
    }

    private VideoTrack createTrack() {
        VideoCapturer capturer = new ScreenCapturerAndroid(mediaProjectionPermissionResultData,
                new MediaProjection.Callback() {
                    @Override
                    public void onStop() {
                        Log.w(TAG, "User revoked permission to capture the screen.");
                    }
                });

        PeerConnectionFactory pcFactory = webRTCModule.mFactory;

        EglBase.Context eglContext = EglUtils.getRootEglBaseContext();
        SurfaceTextureHelper surfaceTextureHelper = SurfaceTextureHelper.create("CaptureThread", eglContext);
        VideoSource source = pcFactory.createVideoSource(capturer.isScreencast());
        capturer.initialize(surfaceTextureHelper, reactContext, source.getCapturerObserver());

        String id = UUID.randomUUID().toString();

        VideoTrack track = pcFactory.createVideoTrack(id, source);
        track.setEnabled(true);

        DisplayMetrics displayMetrics = getDisplayMetrics();
        int width = displayMetrics.widthPixels;
        int height = displayMetrics.heightPixels;
        int fps = 30;
        capturer.startCapture(width, height, fps);

        return track;
    }

    private DisplayMetrics getDisplayMetrics() {
        Activity currentActivity = this.reactContext.getCurrentActivity();
        DisplayMetrics displayMetrics = new DisplayMetrics();
        WindowManager windowManager = (WindowManager) currentActivity.getApplication()
                .getSystemService(Context.WINDOW_SERVICE);
        windowManager.getDefaultDisplay().getRealMetrics(displayMetrics);
        return displayMetrics;
    }
}
