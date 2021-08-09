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

package ai.certifai.faceverification;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;

import ai.certifai.faceverification.pojo.FaceVerifyRequest;
import ai.certifai.faceverification.pojo.FaceVerifyResponse;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

/**
 * Class handling main activity
 *
 * @author willardsm
 */
public class MainActivity extends AppCompatActivity
{
    private static final String[] REQUIRED_PERMISSIONS = new String[]{
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final double FACE_DISTANCE_THRESHOLD = 1.1;
    private static final String API_URL = "http://10.0.2.2:8000";

    ImageView imageView1;
    ImageView imageView2;

    TextView textView;

    Button button;

    ProgressBar progressBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        requestPermissions();

        ActivityResultLauncher<Intent> imageViewActivityResultLauncher1 = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                x -> getImageShowResult(x, imageView1)
        );

        ActivityResultLauncher<Intent> imageViewActivityResultLauncher2 = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                x -> getImageShowResult(x, imageView2)
        );

        imageView1 = findViewById(R.id.imageView);
        imageView2 = findViewById(R.id.imageView2);
        button = findViewById(R.id.button);
        textView = findViewById(R.id.textView);
        progressBar = findViewById(R.id.progressBar);
        progressBar.setVisibility(View.INVISIBLE);

        imageView1.setOnClickListener(view -> selectImage(imageViewActivityResultLauncher1));
        imageView2.setOnClickListener(view -> selectImage(imageViewActivityResultLauncher2));


        button.setOnClickListener(x -> verifyFace(imageView1, imageView2));
    }

    private String getStringImage(ImageView imageView) {
        Bitmap bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();

        ByteArrayOutputStream ba = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 90, ba);
        byte[] by = ba.toByteArray();
        return Base64.encodeToString(by, Base64.DEFAULT);
    }

    private void selectImage(ActivityResultLauncher<Intent> activityResultLauncher) {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);

        activityResultLauncher.launch(Intent.createChooser(intent, "Select Picture"));
    }

    private void getImageShowResult(ActivityResult result, ImageView imageView) {
        if (result.getResultCode() == Activity.RESULT_OK) {
            Intent data = result.getData();
            Uri uri = data.getData();
            imageView.setImageURI(uri);
            imageView.setTag(uri.toString());
        }
    }

    private void verifyFace(ImageView imageView1, ImageView imageView2) {

        if (imageView1.getDrawable() == null || imageView2.getDrawable() == null) {
            Toast.makeText(MainActivity.this, "You need to select an image to proceed",
                    Toast.LENGTH_SHORT).show();
        } else {
            String imageString1 = getStringImage(imageView1);
            String imageString2 = getStringImage(imageView2);

            Retrofit.Builder builder = new Retrofit.Builder()
                    .baseUrl(API_URL)
                    .addConverterFactory(GsonConverterFactory.create());

            Retrofit retrofit = builder.build();

            ApiClient client = retrofit.create(ApiClient.class);

            Call<FaceVerifyResponse> call = client.verifyFace(new FaceVerifyRequest(imageString1, imageString2));
            progressBar.setVisibility(ProgressBar.VISIBLE);
            call.enqueue(new Callback<FaceVerifyResponse>() {
                @Override
                public void onResponse(Call<FaceVerifyResponse> call, Response<FaceVerifyResponse> response) {
                    progressBar.setVisibility(ProgressBar.INVISIBLE);
                    FaceVerifyResponse faceVerifyResponse = response.body();
                    double faceDistance = faceVerifyResponse.getDistance();
                    if (faceDistance > FACE_DISTANCE_THRESHOLD) {
                        textView.setText(String.format("Different Person (Distance: %s)", faceDistance));
                    } else {
                        textView.setText(String.format("Same Person (Distance: %s)", faceDistance));
                    }
                }

                @Override
                public void onFailure(Call<FaceVerifyResponse> call, Throwable t) {
                    progressBar.setVisibility(ProgressBar.INVISIBLE);
                    Toast.makeText(MainActivity.this, "Fail to verify face", Toast.LENGTH_SHORT).show();
                }
            });
        }
    }

    private void requestPermissions() {
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
    }
}