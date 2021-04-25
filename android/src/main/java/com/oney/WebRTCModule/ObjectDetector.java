/*
Based on https://www.tensorflow.org/lite/examples/object_detection/overview
    (https://github.com/tensorflow/examples.git)
    Example code under org.tensorflow.lite.detection is kept unmodified.

Test result with C920 at 1280x720 on TX6:
   Processing in calling thread speed: 3 fps
   Processing in a background thread: 6-7 fps
   Limited to one frame at a time: ~11 fps
   No detection: 15 fps
*/

package com.oney.WebRTCModule;

import java.io.IOException;
import java.util.List;
import java.io.InputStream;
import java.nio.ByteBuffer;

import android.util.Log;
import android.content.res.AssetManager;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.content.Context;
import android.os.Handler;
import android.os.HandlerThread;
import android.opengl.GLES20;

import org.tensorflow.lite.detection.tflite.Detector;
import org.tensorflow.lite.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.webrtc.VideoFrame;
import org.webrtc.VideoFrameDrawer;
import org.webrtc.GlTextureFrameBuffer;
import org.webrtc.NV21Buffer;
import org.webrtc.GlRectDrawer;
import org.webrtc.RendererCommon.GlDrawer;
import org.webrtc.VideoSource;

import org.tensorflow.lite.detection.env.ImageUtils;

public final class ObjectDetector {
    enum DetectorMode {
        TF_OD_API;
    }

    static final String TAG = "ObjectDetector"; // ObjectDetector.class.getCanonicalName();
    static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    static final int TF_INPUT_SIZE = 300;
    static final boolean TF_OD_API_IS_QUANTIZED = true;
    static final DetectorMode MODE = DetectorMode.TF_OD_API;
    static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private Detector detector;
    // weather to save bitmaps onto file system for debugging
    boolean SAVE_BITMAP = false;

    BitmapFrameProcessor bitmapFrameProcessor;
    TextureVideoFrameProcessor textureVideoFrameProcessor;
    NV21VideoFrameProcessor nv21VideoFrameProcessor;

    boolean processingImage = false; // true if a frame is being processed
    private Handler handler;
    private HandlerThread handlerThread;

    public boolean initialize(Context context) {
        try {
            detector = TFLiteObjectDetectionAPIModel.create(
                    context, // assetManager,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_INPUT_SIZE,
                    TF_OD_API_IS_QUANTIZED);
        } catch (final IOException e) {
            Log.e(TAG, "Exception initializing classifier", e);
            // e.printStackTrace();
            return false;
        }

        // copy test image to android/app/src/main/assets/
        // AssetManager assetManager  = context.getAssets();
        // testObjectDetection(assetManager, "tf-test.jpeg");

        return true;
    }

    public void release() {
        if (bitmapFrameProcessor != null) {
            bitmapFrameProcessor.release();
        }
        if (nv21VideoFrameProcessor != null) {
            nv21VideoFrameProcessor.release();
        }
        if (textureVideoFrameProcessor != null) {
            textureVideoFrameProcessor.release();
        }
    }

    /**
     * detect objects in a bitmap image
     *
     * @param bitmap         - the bitmap must be of size required by the model.
     * @param originalWidth  - the original image width
     * @param originalHeight - ht eoriginal image height
     */
    private Rect detect(Bitmap bitmap, final int originalWidth, final int originalHeight, final Matrix transform) {
        RectF peopleRectF = new RectF(0, 0, 0, 0);
        final List<Detector.Recognition> results = detector.recognizeImage(bitmap);
        for (final Detector.Recognition result : results) {
            // Log.d(TAG, result.getTitle() + ": " + result.getConfidence());
            if (result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API && result.getTitle().equals("person")) {
                final RectF location = result.getLocation();
                peopleRectF.union(location.left, location.top, location.right, location.bottom);
                Log.d(TAG, "detected person: confidence=" + result.getConfidence() + ", location=" + location.left + "," + location.top + "," + location.right + ","
                        + location.bottom);
            }
        }

        transform.mapRect(peopleRectF);

        Rect peopleRect = new Rect(0, 0, 0, 0);
        peopleRectF.round(peopleRect);

        // why could they be negative?
        if (peopleRect.left < 0) {
            peopleRect.left = 0;
        }
        if (peopleRect.top < 0) {
            peopleRect.top = 0;
        }
        // floating point rounding may exceeds the original size
        if (peopleRect.right > originalWidth) {
            peopleRect.right = originalWidth;
        }
        if (peopleRect.bottom > originalHeight) {
            peopleRect.bottom = originalHeight;
        }

        Log.d(TAG, String.format("area of interest: %d,%d,%d,%d", peopleRect.left, peopleRect.top, peopleRect.right,
                peopleRect.bottom));

        return peopleRect;
    }

    /**
     * sanity test on object detection.
     *
     * @param assetManager
     * @param file
     */
    private void testObjectDetection(AssetManager assetManager, String file) {
        Log.i(TAG, "test object detection on " + file);
        try {
            InputStream is = assetManager.open(file);
            Bitmap srcBitmap = BitmapFactory.decodeStream(is);
            // Bitmap bitmap = BitmapFactory.decodeFile(pathToPicture);
            Bitmap centeredBitmap;

            if (srcBitmap.getWidth() >= srcBitmap.getHeight()) {
                centeredBitmap = Bitmap.createBitmap(srcBitmap, srcBitmap.getWidth() / 2 - srcBitmap.getHeight() / 2, 0,
                        srcBitmap.getHeight(), srcBitmap.getHeight());

            } else {
                centeredBitmap = Bitmap.createBitmap(srcBitmap, 0, srcBitmap.getHeight() / 2 - srcBitmap.getWidth() / 2,
                        srcBitmap.getWidth(), srcBitmap.getWidth());
            }

            Bitmap croppedBitmap = Bitmap.createBitmap(TF_INPUT_SIZE, TF_INPUT_SIZE, Bitmap.Config.ARGB_8888);
            Matrix frameToCropTransform = ImageUtils.getTransformationMatrix(centeredBitmap.getWidth(),
                    centeredBitmap.getHeight(), TF_INPUT_SIZE, TF_INPUT_SIZE, 0, true);
            Matrix cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);
            final Canvas canvas = new Canvas(croppedBitmap);
            canvas.drawBitmap(centeredBitmap, frameToCropTransform, null);

            Bitmap tfBitmap = centeredBitmap;
            ImageUtils.saveBitmap(croppedBitmap, "tf-test-cropped.png");

            detect(tfBitmap, srcBitmap.getWidth(), srcBitmap.getHeight(), cropToFrameTransform);

            srcBitmap.recycle();
            centeredBitmap.recycle();
            croppedBitmap.recycle();
        } catch (final IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * process a bitmap image as the original source
     */
    class BitmapFrameProcessor {
        Bitmap tfInput; // bitmap as input to Tensorflow model

        int frameWidth = 0; // original video frame width
        int frameHeight = 0; // original video frame height

        Matrix frameToCropTransform;
        Matrix cropToFrameTransform = new Matrix();

        Canvas canvas;

        BitmapFrameProcessor() {
            canvas = new Canvas(tfInput);
            tfInput = Bitmap.createBitmap(TF_INPUT_SIZE, TF_INPUT_SIZE, Bitmap.Config.ARGB_8888);
        }

        void release() {
            if (tfInput != null) {
                tfInput.recycle();
            }
        }

        public Rect process(Bitmap original) {
            // If this is used to handle remote video stream, the size changes frequently
            // as traffic condition changes.
            frameWidth = original.getWidth();
            frameHeight = original.getHeight();

            frameToCropTransform = ImageUtils.getTransformationMatrix(frameWidth, frameHeight, TF_INPUT_SIZE,
                    TF_INPUT_SIZE, 0, true);
            frameToCropTransform.invert(cropToFrameTransform);

            // scale from original size to size expected by TensorFlow model
            canvas.drawBitmap(original, frameToCropTransform, null);

            Rect rect = detect(tfInput, frameWidth, frameHeight, cropToFrameTransform);

            if (SAVE_BITMAP) {
                ImageUtils.saveBitmap(original, "original.png");
                ImageUtils.saveBitmap(tfInput, "input.png");
                if (rect != null && !rect.isEmpty()) {
                    Bitmap detectedImage = Bitmap.createBitmap(original, rect.left, rect.top, rect.right - rect.left,
                            rect.bottom - rect.top);
                    ImageUtils.saveBitmap(detectedImage, "detected.png");
                    detectedImage.recycle();
                }
                SAVE_BITMAP = false;
            }

            return rect;
        }
    }

    /**
     * Detect objects in a Bitmap image.
     * <p>
     * WebRTC EglRenderer provides video frames in Bitmap format. This is used when object
     * detection is performed on remote video streams.
     *
     * @param image
     * @return
     */
    public Rect processBitmap(final Bitmap image) {
        if (bitmapFrameProcessor == null) {
            bitmapFrameProcessor = new BitmapFrameProcessor();
        }

        return bitmapFrameProcessor.process(image);
    }

    /**
     * Texture VideoFrame handling
     */
    class TextureVideoFrameProcessor {
        private VideoFrameDrawer videoFrameDrawer = null;
        private GlDrawer glDrawer = null;
        private GlTextureFrameBuffer bitmapTextureFramebuffer = null;
        private Matrix drawMatrix;
        private final ByteBuffer bitmapBuffer = ByteBuffer.allocateDirect(TF_INPUT_SIZE * TF_INPUT_SIZE * 4);
        final Bitmap tfInput;

        // video frame width and height, and mapping to and frame cropped.
        int frameWidth;
        int frameHeight;

        // It's called crop, but currently it's scaling. Scaling distorts the image fed into
        // Tensorflow. This may affect the detection result. Cropping may be more suitable.
        Matrix frameToCropTransform;
        Matrix cropToFrameTransform = new Matrix();

        final int scale = 8;
        Bitmap scaledOriginal;

        TextureVideoFrameProcessor() {
            glDrawer = new GlRectDrawer();
            videoFrameDrawer = new VideoFrameDrawer();

            tfInput = Bitmap.createBitmap(TF_INPUT_SIZE, TF_INPUT_SIZE, Bitmap.Config.ARGB_8888);

            drawMatrix = new Matrix();
            drawMatrix.reset();
            drawMatrix.preTranslate(0.5f, 0.5f);
            drawMatrix.preScale(1f, -1f); // image is upside down
            drawMatrix.preTranslate(-0.5f, -0.5f);

            bitmapTextureFramebuffer = new GlTextureFrameBuffer(GLES20.GL_RGBA);
        }

        void release() {
            videoFrameDrawer.release();
            glDrawer.release();
            if (bitmapTextureFramebuffer != null) bitmapTextureFramebuffer.release();
            tfInput.recycle();
        }

        Bitmap drawTextureBufferToBitmap(final VideoFrame frame, Bitmap dst) {
            final int bitmapWidth = dst.getWidth();
            final int bitmapHeight = dst.getHeight();

            // java.lang.IllegalStateException: Framebuffer not complete, status: 0
            // This was caused by calling this with the same buffer by two different threads.
            // GlTextureFrameBuffer can't be used by multiple threads.
            // Log.d(TAG, "thread " + Thread.currentThread().getId());
            bitmapTextureFramebuffer.setSize(bitmapWidth, bitmapHeight);

            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, bitmapTextureFramebuffer.getFrameBufferId());
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D,
                    bitmapTextureFramebuffer.getTextureId(), 0);

            GLES20.glClearColor(0 /* red */, 0 /* green */, 0 /* blue */, 0 /* alpha */);
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
            videoFrameDrawer.drawFrame(frame, glDrawer, drawMatrix, 0 /* viewportX */, 0 /* viewportY */, bitmapWidth,
                    bitmapHeight);

            bitmapBuffer.rewind();
            GLES20.glViewport(0, 0, bitmapWidth, bitmapHeight);
            GLES20.glReadPixels(0, 0, bitmapWidth, bitmapHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, bitmapBuffer);

            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
            int error = GLES20.glGetError();
            if (error != GLES20.GL_NO_ERROR) {
                Log.d(TAG, "GLES20 error " + error);
                return null;
            }

            dst.copyPixelsFromBuffer(bitmapBuffer);

            return dst;
        }

        /**
         * Copy VideoFrame with texture buffer into a bitmap.
         * <p>
         * EglRenderer.java has an example of drawing texture frame buffer into a
         * bitmap. Below follows that example.
         * <p>
         * sdk/android/api/org/webrtc/YuvConverter.java can convert from OES texture to
         * I420 YUV. It's possible to take advantage of this.
         *
         * @param frame
         */
        public Boolean copy(VideoFrame frame) {
            VideoFrame.TextureBuffer buffer = (VideoFrame.TextureBuffer) frame.getBuffer();
            final int width = buffer.getWidth();
            final int height = buffer.getHeight();
            Log.d(TAG, "onFrame " + buffer.getType() + " w=" + width + "h=" + height);
            // if (true) return null;

            // Reconfigure if video changes size. This should happen only on the first frame
            // when no configuration exists. In other words, video size shouldn't change
            // middle of a session. One possible exception may be when rotation takes place.
            if (width != frameWidth || height != frameHeight) {
                frameWidth = width;
                frameHeight = height;
                frameToCropTransform = ImageUtils.getTransformationMatrix(width, height, TF_INPUT_SIZE, TF_INPUT_SIZE,
                        0, true);

                // TODO: fix coordinate mapping from detected rect to original image
                // VideoFrameDrawer performs some transformation. Below attempts to capture that
                // too. However below doesn't produce correct result.
                Matrix transformSum = new Matrix();
                transformSum.preTranslate(0.5f, 0.5f);
                transformSum.preRotate(frame.getRotation());
                transformSum.preTranslate(-0.5f, -0.5f);
                transformSum.preConcat(drawMatrix);
                transformSum.postConcat(frameToCropTransform);
                transformSum.invert(cropToFrameTransform);
            }

            if (drawTextureBufferToBitmap(frame, tfInput) == null) {
                return false;
            }

            if (SAVE_BITMAP) {
                // Save the original

                // createBitmap causes: Fatal signal 11 (SIGSEGV), code 2 (SEGV_ACCERR)
                // Can't be caught either. likely cause is not enough of memory.
                // try {
                //     Bitmap original = Bitmap.createBitmap(width(), height, Bitmap.Config.ARGB_8888);
                //     if (original != null && textureVideoFrameDrawer.drawTextureBufferToBitmap(frame, original) != null) {
                //         ImageUtils.saveBitmap(original, "original.png");
                //         original.recycle();
                //     } else {
                //         Log.d(TAG, "failed to create original");
                //     }
                // } catch (final Throwable e) {
                //     Log.e(TAG, "createBitmap failed", e);
                // }

                // The exact original image is not essential. A small image with aspect ratio
                // preserved would tell whether detection is correct.
                scaledOriginal = Bitmap.createBitmap(frameWidth / scale, frameHeight / scale, Bitmap.Config.ARGB_8888);
                if (scaledOriginal == null) {
                    Log.d(TAG, "cannot create bitmap to save the original");
                } else if (drawTextureBufferToBitmap(frame, scaledOriginal) == null) {
                    Log.d(TAG, "cannot draw texture buffer to save the original");
                } else {
                    ImageUtils.saveBitmap(scaledOriginal, "scaled-original.png");
                }
                if (scaledOriginal != null) {
                    scaledOriginal.recycle();
                }

            }

            return true;
        }

        public Rect process() {
            Rect rect = detect(tfInput, frameWidth, frameHeight, cropToFrameTransform);

            if (SAVE_BITMAP) {
                if (rect != null && !rect.isEmpty()) {
                    Bitmap detectedImage = Bitmap.createBitmap(scaledOriginal, rect.left / scale, rect.top / scale,
                            (rect.right - rect.left) / scale, (rect.bottom - rect.top) / scale);
                    ImageUtils.saveBitmap(detectedImage, "detected.png");
                    detectedImage.recycle();
                }

                ImageUtils.saveBitmap(tfInput, "input.png");

                SAVE_BITMAP = false;
            }

            return rect;
        }
    }

    /**
     * Process a NV21 VideoFrame and return area of detected objects.
     * <p>
     * Legacy camera API onPreviewFrame uses YUV420SP(aka YCbCr_420_SP, or NV21)
     */
    class NV21VideoFrameProcessor {
        Bitmap original = null; // bitmap representing original image
        Bitmap tfInput; // bitmap as input to Tensorflow model

        // RGB byte array and its size (width x height)
        int[] rgbBytes = null;

        int frameWidth = 0; // original video frame width
        int frameHeight = 0; // original video frame height

        Matrix frameToCropTransform;
        Matrix cropToFrameTransform = new Matrix();

        Canvas canvas;

        NV21VideoFrameProcessor() {
            tfInput = Bitmap.createBitmap(TF_INPUT_SIZE, TF_INPUT_SIZE, Bitmap.Config.ARGB_8888);
            canvas = new Canvas(tfInput);
        }

        void release() {
            if (original != null) {
                original.recycle();
            }
            if (tfInput != null) {
                tfInput.recycle();
            }
        }

        public boolean extractImage(NV21Buffer buffer) {
            // create working buffer and bitmap
            if (rgbBytes == null || frameWidth != buffer.getWidth() || frameHeight != buffer.getHeight()) {
                frameWidth = buffer.getWidth();
                frameHeight = buffer.getHeight();
                rgbBytes = new int[frameWidth * frameHeight];
                if (rgbBytes == null) {
                    Log.d(TAG, "cannot allocate buffer");
                    return false;
                }

                original = Bitmap.createBitmap(frameWidth, frameHeight, Bitmap.Config.ARGB_8888);
                if (original == null) {
                    Log.d(TAG, "cannot create bitmap");
                    return false;
                }

                frameToCropTransform = ImageUtils.getTransformationMatrix(frameWidth, frameHeight, TF_INPUT_SIZE,
                        TF_INPUT_SIZE, 0, true);
                frameToCropTransform.invert(cropToFrameTransform);
            }

            ImageUtils.convertYUV420SPToARGB8888(buffer.getData(), frameWidth, frameHeight, rgbBytes);
            original.setPixels(rgbBytes, 0, frameWidth, 0, 0, frameWidth, frameHeight);

            // scale from original size to size expected by TensorFlow model
            canvas.drawBitmap(original, frameToCropTransform, null);

            return true;
        }

        // Process a NV21 VideoFrame and return area of detected objects.
        public Rect process() {
            Rect rect = detect(tfInput, frameWidth, frameHeight, cropToFrameTransform);

            if (SAVE_BITMAP) {
                ImageUtils.saveBitmap(original, "original.png");
                ImageUtils.saveBitmap(tfInput, "cropped.png");
                if (rect != null && !rect.isEmpty()) {
                    Bitmap detectedImage = Bitmap.createBitmap(original, rect.left, rect.top, rect.right - rect.left,
                            rect.bottom - rect.top);
                    ImageUtils.saveBitmap(detectedImage, "detected.png");
                    detectedImage.recycle();
                }
                SAVE_BITMAP = false;
            }

            return rect;
        }
    }

    public void processVideoFrameInBackground(VideoFrame frame, VideoSource source) {
        if (processingImage || handler == null) return;

        // Extract image from the the video frame. This must be
        // done on the calling thread. Once return to caller
        // the video frame should not be touched.
        if (frame.getBuffer() instanceof VideoFrame.TextureBuffer) {
            if (textureVideoFrameProcessor == null) {
                textureVideoFrameProcessor = new TextureVideoFrameProcessor();
            }
            if (textureVideoFrameProcessor.copy(frame) == false) return;
        } else if (frame.getBuffer() instanceof NV21Buffer) {
            if (nv21VideoFrameProcessor == null) {
                nv21VideoFrameProcessor = new NV21VideoFrameProcessor();
            }
            NV21Buffer nv21Buffer = (NV21Buffer) frame.getBuffer();
            nv21VideoFrameProcessor.extractImage(nv21Buffer);
        } else {
            Log.d(TAG, "unsupported VideoFrame");
            return;
        }

        // detection is done in a separate thread
        processingImage = true;
        Log.d(TAG, "processing");

        handler.post(new Runnable() {
            @Override
            public void run() {
                Rect rect = null;
                if (frame.getBuffer() instanceof VideoFrame.TextureBuffer) {
                    rect = textureVideoFrameProcessor.process();
                } else if (frame.getBuffer() instanceof NV21Buffer) {
                    rect = nv21VideoFrameProcessor.process();
                }

                if (rect != null) {
                    source.setCrop(rect);
                }
                Log.d(TAG, "processing done");
                processingImage = false;
            }
        });
    }

    void resume() {
        if (handlerThread == null) {
            handlerThread = new HandlerThread("inference");
            handlerThread.start();
        }
        if (handler == null) {
            handler = new Handler(handlerThread.getLooper());
        }
    }

    void suspend() {
        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            Log.e(TAG, "InterruptedException");
        }
    }
}