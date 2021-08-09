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

package ai.certifai.fruitclassifier;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.certifai.fruitclassifier.model.FruitClassifier;
import ai.certifai.fruitclassifier.utils.AssetHandler;
import ai.certifai.fruitclassifier.utils.ImageReader;

/**
 * Class handling live view prediction
 */
public class LiveViewActivity extends AppCompatActivity
{
    // layouts
    private TextView result;
    private PreviewView liveView;

    private boolean isLandscape;

    // camera variables
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ImageAnalysis analysis;
    private ExecutorService cameraExecutor;

    // model
    private FruitClassifier fruitClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_view);

        isLandscape = this.getIntent().getBooleanExtra("isLandscape", false);

        // define layout
        liveView = findViewById(R.id.liveView);
        setUpLiveView();
        result = findViewById(R.id.liveResult);

        // load model
        try
        {
            fruitClassifier = new FruitClassifier(AssetHandler.assetFilePath(this, "model.pt"));
        }
        catch (IOException e)
        {
            Log.e("FruitClassifier", "Error reading assets", e);
            finish();
        }

        // define new thread to run analysis
        cameraExecutor = Executors.newSingleThreadExecutor();

        // set up camera
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(this::listenToCameraProvider, ContextCompat.getMainExecutor(this));
    }

    private void setUpLiveView()
    {
        int rotation = isLandscape ? 270 : 0;
        liveView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        liveView.setRotation(rotation);
    }

    private void analyze(ImageProxy image) {
        int rotation = isLandscape ? 0 : 90;
        Bitmap imageBitmap = ImageReader.YUVToBitmap(image, rotation);

        if (imageBitmap != null)
        {
            String cls = fruitClassifier.predict(imageBitmap);

            Runnable postResult = () -> this.setResultClass(cls);
            liveView.post(postResult);
        }

        image.close();
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
        analysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        analysis.setAnalyzer(cameraExecutor, this::analyze);

        // image preview
        Preview preview = new Preview.Builder().build();

        preview.setSurfaceProvider(liveView.getSurfaceProvider());

        // fix to back camera
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        cameraProvider.bindToLifecycle(this, cameraSelector, analysis, preview);
    }

    public void setResultClass(String cls)
    {
        result.setText(cls);
        result.setVisibility(View.VISIBLE);
    }

    @Override
    protected void onDestroy()
    {
        super.onDestroy();
        cameraExecutor.shutdown();
    }
}