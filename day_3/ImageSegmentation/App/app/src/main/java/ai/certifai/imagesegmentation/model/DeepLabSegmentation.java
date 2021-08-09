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

package ai.certifai.imagesegmentation.model;

import android.graphics.Bitmap;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Model used in app
 *
 * @author willardsm
 */
public class DeepLabSegmentation
{
    private static final float[] NORM_MEAN = new float[]{0.485f, 0.456f, 0.406f};
    private static final float[] NORM_STD = new float[]{0.229f, 0.224f, 0.225f};
    private static final int CLASSNUM = 21;

    // pixel indices correspond to classes in alphabetical order (1=aeroplane, 2=bicycle, 3=bird,
    // 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog, 13=horse,
    // 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor)
    private static final int OTHERS = 0;
    private static final int AEROPLANE = 1;
    private static final int BICYCLE = 2;
    private static final int BIRD = 3;
    private static final int BOAT = 4;
    private static final int BOTTLE = 5;
    private static final int BUS = 6;
    private static final int CAR = 7;
    private static final int CAT = 8;
    private static final int CHAIR = 9;
    private static final int COW = 10;
    private static final int DININGTABLE = 11;
    private static final int DOG = 12;
    private static final int HORSE = 13;
    private static final int MOTORBIKE = 14;
    private static final int PERSON = 15;
    private static final int POTTED_PLANT = 16;
    private static final int SHEEP = 17;
    private static final int SOFA = 18;
    private static final int TRAIN = 19;
    private static final int TV_OR_MONITOR = 20;
    private final Module module;
    Map<Integer, Integer> colors = new HashMap<>();

    public DeepLabSegmentation(String modelPath)
    {
        module = Module.load(modelPath);
        colors.put(OTHERS, 0xff000000);
        colors.put(AEROPLANE, 0xff9e5c1a);
        colors.put(BICYCLE, 0xff9e7d1a);
        colors.put(BIRD, 0xff9e9e1a);
        colors.put(BOAT, 0xff7d9e1a);
        colors.put(BOTTLE, 0xff5c9e1a);
        colors.put(BUS, 0xff3b9e1a);
        colors.put(CAR, 0xff1a9e1a);
        colors.put(CAT, 0xff1a9e3b);
        colors.put(CHAIR, 0xff1a9e51);
        colors.put(COW, 0xff1a9e72);
        colors.put(DININGTABLE, 0xff1a9e93);
        colors.put(DOG, 0xff1a889e);
        colors.put(HORSE, 0xff1a679e);
        colors.put(MOTORBIKE, 0xff1a469e);
        colors.put(PERSON, 0xff1a259e);
        colors.put(POTTED_PLANT, 0xff301a9e);
        colors.put(SHEEP, 0xff511a9e);
        colors.put(SOFA, 0xff721a9e);
        colors.put(TRAIN, 0xff931a9e);
        colors.put(TV_OR_MONITOR, 0xff9e1a88);
    }

    public Bitmap predict(Bitmap imageBitmap)
    {
        Bitmap croppedImage = Bitmap.createScaledBitmap(imageBitmap, 224, 224, false);

        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(croppedImage, NORM_MEAN, NORM_STD);
        Map<String, IValue> outputDict = module.forward(IValue.from(inputTensor)).toDictStringKey();
        final Tensor outputTensor = Objects.requireNonNull(outputDict.get("out")).toTensor();
        final float[] scores = outputTensor.getDataAsFloatArray();
        int width = croppedImage.getWidth();
        int height = croppedImage.getHeight();
        int[] intValues = new int[width * height];

        for (int j = 0; j < width; j++) {
            for (int k = 0; k < height; k++) {
                int maxi = 0, maxj = 0, maxk = 0;
                double maxnum = -Double.MAX_VALUE;
                for (int i = 0; i < CLASSNUM; i++) {
                    if (scores[i * (width * height) + j * width + k] > maxnum) {
                        maxnum = scores[i * (width * height) + j * width + k];
                        maxi = i;
                        maxj = j;
                        maxk = k;
                    }
                }
                intValues[maxj * width + maxk] = colors.get(maxi);
            }
        }

        Bitmap bmpSegmentation = Bitmap.createScaledBitmap(croppedImage, width, height, true);
        Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
        outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());

        return Bitmap.createScaledBitmap(outputBitmap, imageBitmap.getWidth(), imageBitmap.getHeight(), true);
    }
}
