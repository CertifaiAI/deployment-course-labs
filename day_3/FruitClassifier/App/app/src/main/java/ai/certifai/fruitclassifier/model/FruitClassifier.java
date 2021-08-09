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

package ai.certifai.fruitclassifier.model;

import android.graphics.Bitmap;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

/**
 * Model used in app
 *
 * @author YCCertifai
 */
public class FruitClassifier
{
    private Module module;
    private static final float[] NORM_MEAN = new float[] {0.485f, 0.456f, 0.406f};
    private static final float[] NORM_STD = new float[] {0.229f, 0.224f, 0.225f};
    private static final String[] CLASSES = {"apple", "grapes", "lemon"};

    public FruitClassifier(String modelPath)
    {
        module = Module.load(modelPath);
    }

    public String predict(Bitmap imageBitmap)
    {
        Bitmap croppedImage = Bitmap.createScaledBitmap(imageBitmap, 150, 150, false);
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(croppedImage, NORM_MEAN, NORM_STD);
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        float[] scores = outputTensor.getDataAsFloatArray();

        int maxIdx = getMaxIdx(scores);

        return CLASSES[maxIdx];
    }

    private int getMaxIdx(float[] scores)
    {
        float maxScore = -Float.MAX_VALUE;
        int maxIdx = -1;
        for (int i = 0; i < scores.length; i++)
        {
            if (scores[i] > maxScore)
            {
                maxScore = scores[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
