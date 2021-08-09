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
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.IOException;

import ai.certifai.fruitclassifier.model.FruitClassifier;
import ai.certifai.fruitclassifier.utils.AssetHandler;
import ai.certifai.fruitclassifier.utils.ImageReader;

/**
 * Class handling result showing activity
 *
 * @author YCCertifai
 */
public class ResultActivity extends AppCompatActivity
{
    // layouts
    private ImageView image;
    private TextView clsView;

    private String fileLoc;

    // model
    private FruitClassifier fruitClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        // define layouts
        image = findViewById(R.id.resultView);
        clsView = findViewById(R.id.textView);

        // load model
        try
        {
            fruitClassifier = new FruitClassifier(AssetHandler.assetFilePath(this, "model.pt"));
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
        String cls = fruitClassifier.predict(imageBitmap);
        Runnable postCls = () -> this.setResult(cls);
        clsView.post(postCls);
    }

    private File getImageFile()
    {
        fileLoc = getIntent().getStringExtra("image_file");
        return new File(fileLoc);
    }

    private void setResult(String cls)
    {
        clsView.setText(cls);
        clsView.setVisibility(View.VISIBLE);
    }

    private void setImage(Bitmap imageBitmap)
    {
        image.setImageBitmap(imageBitmap);
        image.setVisibility(View.VISIBLE);
    }
}