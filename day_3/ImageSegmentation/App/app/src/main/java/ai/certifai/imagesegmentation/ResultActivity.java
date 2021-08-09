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

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.ProgressBar;

import java.io.File;
import java.io.IOException;

import ai.certifai.imagesegmentation.model.DeepLabSegmentation;
import ai.certifai.imagesegmentation.utils.AssetHandler;
import ai.certifai.imagesegmentation.utils.ImageReader;

/**
 * Class handling result showing activity
 *
 * @author willadrsm
 */
public class ResultActivity extends AppCompatActivity implements Runnable
{
    ImageView resultView;
    ImageView imageView;
    ProgressBar progressBar;

    Bitmap imageBitmap;

    DeepLabSegmentation deepLabSegmentation;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        imageView = findViewById(R.id.imageView);
        resultView = findViewById(R.id.resultView);
        progressBar = findViewById(R.id.progressBar);

        progressBar.setVisibility(ProgressBar.VISIBLE);

        File imageFile = getImageFile();
        imageBitmap = ImageReader.readFileToImage(imageFile);
        imageView.setImageBitmap(imageBitmap);

        Thread thread = new Thread(ResultActivity.this);
        thread.start();

    }

    private File getImageFile()
    {
        String fileLoc = getIntent().getStringExtra("image_file");
        return new File(fileLoc);
    }

    @Override
    public void run()
    {
        try
        {
            deepLabSegmentation = new DeepLabSegmentation(AssetHandler.assetFilePath(this, "model.pt"));
        }
        catch (IOException e)
        {
            Log.e("DeepLab Segmentation", "Error reading assets", e);
            finish();
        }
        Bitmap output = deepLabSegmentation.predict(imageBitmap);
        runOnUiThread(() ->
                {
                    resultView.setImageBitmap(output);
                    progressBar.setVisibility(ProgressBar.INVISIBLE);
                }
        );
    }
}