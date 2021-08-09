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

package ai.certifai.imagesegmentation;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Arrays;

import ai.certifai.imagesegmentation.utils.FilePath;

/**
 * Class handling main activity
 *
 * @author willardsm
 */
public class MainActivity extends AppCompatActivity
{
    // permission
    private static final String[] REQUIRED_PERMISSIONS = new String[]{
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private static final int REQUEST_CODE_PERMISSIONS = 10;

    // layout
    private Button enableCamera;
    private Button selectImage;
    private SwitchCompat toggle;
    private TextView toggleText;

    private ActivityResultLauncher<Intent> chooseImageFileLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // get result from file choosing
        chooseImageFileLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                this::getImageShowResult);

        // define layouts
        enableCamera = findViewById(R.id.enableCamera);
        selectImage = findViewById(R.id.chooseImage);
        toggle = findViewById(R.id.switchToggle);
        toggleText = findViewById(R.id.switchText);

        requestPermissions();

        // set button functions
        enableCamera.setOnClickListener(x -> openCamera());
        selectImage.setOnClickListener(x -> selectImage());
        toggle.setOnCheckedChangeListener((v, c) -> changeText(c));
    }

    private void changeText(boolean isChecked)
    {
        toggleText.setText(isChecked ? toggle.getTextOn() : toggle.getTextOff());
    }

    private void getImageShowResult(ActivityResult result)
    {
        if (result.getResultCode() == Activity.RESULT_OK) {
            Intent data = result.getData();
            Uri uri = data.getData();
            String selectedFilePath = FilePath.getPath(MainActivity.this, uri);
            Intent intent = new Intent(this, ResultActivity.class);
            intent.putExtra("image_file", selectedFilePath);

            startActivity(intent);
        }
    }

    private void openCamera()
    {
        Intent intent = new Intent(this, CameraActivity.class);
        intent.putExtra("isLandscape", toggle.isChecked());
        startActivity(intent);
    }

    private void selectImage()
    {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        chooseImageFileLauncher.launch(intent);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS && !hasCameraPermission()) {
            Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    private void requestPermissions()
    {
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
    }

    private boolean hasCameraPermission()
    {
        return Arrays.stream(REQUIRED_PERMISSIONS)
                .map(permission -> ContextCompat.checkSelfPermission(this, permission))
                .allMatch(res -> res == PackageManager.PERMISSION_GRANTED);
    }
}