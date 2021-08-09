/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
 * This program is part of OSRFramework. You can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version. You should have received a copy of the
 * GNU Affero General Public License along with this program.  If not, see
 * https://www.gnu.org/licenses/agpl-3.0
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 */

package ai.certifai.yolov5;

import android.graphics.Bitmap;
import android.hardware.camera2.CaptureRequest;
import android.os.Bundle;
import android.util.Log;
import android.util.Range;
import android.view.View;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.camera2.interop.Camera2Interop;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.certifai.yolov5.model.Yolov5;
import ai.certifai.yolov5.utils.AssetHandler;
import ai.certifai.yolov5.utils.ImageReader;

/**
 * Class handling live view prediction
 *
 * @author YCCertifai
 */
public class LiveViewActivity extends AppCompatActivity
{
    // layouts
    private ImageView liveView;
    private boolean isLandscape;

    // camera variables
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ImageAnalysis analysis;
    private ExecutorService cameraExecutor;

    // model
    private Yolov5 yolov5;
    private double confThreshold = 0.5;
    private double iouThreshold = 0.5;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_view);

        isLandscape = this.getIntent().getBooleanExtra("isLandscape", false);

        // define layout
        liveView = findViewById(R.id.liveView);

        // load model
        try
        {
            yolov5 = new Yolov5(AssetHandler.assetFilePath(this, "model.pt"));
        } catch (IOException e) {
            Log.e("FruitClassifier", "Error reading assets", e);
            finish();
        }

        // define new thread to run analysis
        cameraExecutor = Executors.newSingleThreadExecutor();

        // set up camera
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(this::listenToCameraProvider, ContextCompat.getMainExecutor(this));
    }

    private void analyze(ImageProxy image)
    {
        int rotation = isLandscape ? 0 : 90;
        Bitmap imageBitmap = ImageReader.YUVToBitmap(image, rotation);

        if (imageBitmap != null)
        {
            Bitmap outputImage = imageBitmap;
            try
            {
                outputImage = yolov5.predict(imageBitmap, confThreshold, iouThreshold);
            } catch (Exception ignore) {}

            Bitmap finalOutputImage = outputImage;

            Runnable postResult = () -> this.setLiveView(finalOutputImage);
            liveView.post(postResult);
        }

        image.close();
    }

    private void setLiveView(Bitmap image)
    {
        liveView.setImageBitmap(image);
        liveView.setVisibility(View.VISIBLE);
    }

    private void listenToCameraProvider()
    {
        try
        {
            ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
            bindImageAnalysis(cameraProvider);
        }
        catch (ExecutionException | InterruptedException e)
        {
            e.printStackTrace();
        }
    }

    private void bindImageAnalysis(ProcessCameraProvider cameraProvider)
    {
        // image analysis -> model prediction run here
        ImageAnalysis.Builder builder = new ImageAnalysis.Builder();
        Camera2Interop.Extender<ImageAnalysis> ext = new Camera2Interop.Extender<>(builder);
        ext.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<Integer>(10,15));
        analysis = builder.setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        analysis.setAnalyzer(cameraExecutor, this::analyze);



        // fix to back camera
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        cameraProvider.bindToLifecycle(this, cameraSelector, analysis);
    }

    @Override
    protected void onDestroy()
    {
        super.onDestroy();
        cameraExecutor.shutdown();
    }
}