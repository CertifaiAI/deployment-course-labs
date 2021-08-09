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

import android.content.Intent;
import android.os.Bundle;
import android.view.Surface;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

/**
 * Class handling camera activity
 *
 * @author YCCertifai
 */
public class CameraActivity extends AppCompatActivity
{
    // layout
    private PreviewView previewView;
    private Button takePicture;
    private boolean isLandscape;

    // camera variables
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ImageCapture imageCapture;

    private static File outputDirectory;
    private static final String FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS";

    private File photoFile;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        // get orientation
        isLandscape = this.getIntent().getBooleanExtra("isLandscape", false);

        // define layout
        previewView = findViewById(R.id.previewView);
        setUpPreviewView();
        takePicture = findViewById(R.id.takePicture);

        outputDirectory = getOutputDirectory();

        // set up camera
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(this::listenToCameraProvider, ContextCompat.getMainExecutor(this));

        // set buttons functions
        takePicture.setOnClickListener(v -> onClick());
    }

    private void setUpPreviewView()
    {
        int rotation = isLandscape ? 270 : 0;
        previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        previewView.setRotation(rotation);
    }

    private void listenToCameraProvider()
    {
        try
        {
            ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
            bindImageCapture(cameraProvider);
        }
        catch (ExecutionException | InterruptedException e)
        {
            e.printStackTrace();
        }
    }

    private File getOutputDirectory()
    {
        File mediaDir = Arrays.stream(getExternalMediaDirs())
                .findFirst()
                .orElse(getFilesDir());

        if (!mediaDir.equals(getFilesDir()) && !mediaDir.exists())
        {
            mediaDir.mkdir();
        }

        return mediaDir;
    }

    private void bindImageCapture(ProcessCameraProvider cameraProvider)
    {
        // image capture -> photo taking logic run here
        imageCapture = new ImageCapture.Builder()
                .setTargetRotation(setRotation())
                .build();

        // image preview
        Preview preview = new Preview.Builder().build();


        preview.setSurfaceProvider(previewView.getSurfaceProvider());


        // set to back camera
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        cameraProvider.bindToLifecycle(this, cameraSelector, imageCapture, preview);
    }

    private int setRotation()
    {
        return isLandscape ? Surface.ROTATION_90 : Surface.ROTATION_0;
    }

    private void onClick()
    {
        photoFile = new File(outputDirectory,
                new SimpleDateFormat(FILENAME_FORMAT, Locale.US).format(System.currentTimeMillis()) + ".jpg");

        ImageCapture.OutputFileOptions outputFileOptions = new ImageCapture.OutputFileOptions
                .Builder(photoFile)
                .build();

        imageCapture.takePicture(outputFileOptions, ContextCompat.getMainExecutor(this), new CallBack());
    }

    private void toResultActivity()
    {
        Intent intent = new Intent(this, ResultActivity.class);
        intent.putExtra("image_file", photoFile.getAbsolutePath());
        startActivity(intent);
    }

    // callback class for image capture
    private class CallBack implements ImageCapture.OnImageSavedCallback
    {
        @Override
        public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults)
        {
            Toast.makeText(CameraActivity.this, "Image capture successful", Toast.LENGTH_SHORT).show();
            toResultActivity();
        }

        @Override
        public void onError(@NonNull ImageCaptureException exception)
        {
            Toast.makeText(CameraActivity.this, "Image capture failed", Toast.LENGTH_SHORT).show();
        }
    }
}