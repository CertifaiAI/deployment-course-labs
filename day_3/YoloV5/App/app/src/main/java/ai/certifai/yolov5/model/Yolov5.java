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

package ai.certifai.yolov5.model;

import android.graphics.Bitmap;
import android.graphics.Color;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.util.List;

import ai.certifai.yolov5.model.utils.ImageProcessing;
import ai.certifai.yolov5.model.utils.ResultProcessing;

/**
 * Model used by the app
 *
 * @author YCCertifai
 */
public class Yolov5
{
    private Module module;
    private static final float[] NORM_MEAN = new float[] {0, 0, 0};
    private static final float[] NORM_STD = new float[] {1, 1, 1};
    private static final int[] IMAGE_SIZE = new int[] {640, 640};
    public Yolov5(String assetFilePath)
    {
        module = Module.load(assetFilePath);
        YoloResult.setImageSize(IMAGE_SIZE);
    }

    public Bitmap predict(Bitmap imageBitmap, double confThreshold, double iouThreshold)
    {
        Bitmap processedImage = ImageProcessing.letterbox(imageBitmap, IMAGE_SIZE, Color.GRAY);
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(processedImage, NORM_MEAN, NORM_STD);
        IValue[] outputTuple = module.forward(IValue.from(inputTensor)).toTuple();

        Tensor outputTensor = outputTuple[0].toTensor();
        List<YoloResult> output = ResultProcessing.processResult(outputTensor, confThreshold, iouThreshold);

        return ImageProcessing.drawResult(imageBitmap, output, IMAGE_SIZE);
    }
}
