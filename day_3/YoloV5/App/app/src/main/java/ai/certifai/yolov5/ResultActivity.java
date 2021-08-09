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
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.IOException;

import ai.certifai.yolov5.model.Yolov5;
import ai.certifai.yolov5.utils.AssetHandler;
import ai.certifai.yolov5.utils.ImageReader;

/**
 * Class handling result activity
 *
 * @author YCCertifai
 */
public class ResultActivity extends AppCompatActivity
{
    // layouts
    private ImageView image;

    private String fileLoc;

    // model
    private Yolov5 yolov5;
    private double confThreshold = 0.5;
    private double iouThreshold = 0.5;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        // define layouts
        image = findViewById(R.id.resultView);

        // load model
        try
        {
            yolov5 = new Yolov5(AssetHandler.assetFilePath(this, "model.pt"));
        } catch (IOException e)
        {
            Log.e("FruitClassifier", "Error reading assets", e);
            finish();
        }

        // load image
        File imageFile = getImageFile();
        Bitmap imageBitmap = ImageReader.readFileToImage(imageFile);

        setImage(imageBitmap);

        // predict
        Runnable predictionRunnable = () -> runPrediction(imageBitmap);
        new Thread(predictionRunnable).start();
    }

    private void runPrediction(Bitmap imageBitmap)
    {
        Bitmap box = yolov5.predict(imageBitmap, confThreshold, iouThreshold);
        Runnable postImage = () -> this.setImage(box);
        image.post(postImage);
    }

    private File getImageFile()
    {
        fileLoc = getIntent().getStringExtra("image_file");
        return new File(fileLoc);
    }

    private void setImage(Bitmap imageBitmap)
    {
        image.setImageBitmap(imageBitmap);
        image.setVisibility(View.VISIBLE);
    }
}