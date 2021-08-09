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

package ai.certifai.yolov5.model.utils;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;

import java.util.List;

import ai.certifai.yolov5.model.CocoClasses;
import ai.certifai.yolov5.model.YoloResult;

/**
 * Class to perform image processing in YOLOv5
 *
 * @author YCCertifai
 */
public class ImageProcessing
{
    public static Bitmap letterbox(Bitmap image, int[] imageSize, int backgroundColor)
    {
        int width = image.getWidth();
        int height = image.getHeight();

        float scaleRatio = Math.min((float)imageSize[0] / width, (float)imageSize[1] / height);

        int newWidth = (int) (width * scaleRatio);
        int newHeight = (int) (height * scaleRatio);

        int left = (imageSize[0] - newWidth) / 2;
        int top = (imageSize[1] - newHeight) / 2;

        Bitmap scaledImage = Bitmap.createScaledBitmap(image, newWidth, newHeight, false);

        Bitmap imageWithBorder = Bitmap.createBitmap(imageSize[0], imageSize[1], image.getConfig());

        Canvas canvas = new Canvas(imageWithBorder);
        canvas.drawColor(backgroundColor);
        canvas.drawBitmap(scaledImage, left, top, null);

        return imageWithBorder;
    }


    public static Bitmap drawResult(Bitmap imageBitmap, List<YoloResult> output, int[] imageSize)
    {
        Bitmap outputBitmap = imageBitmap.copy(imageBitmap.getConfig(), true);
        Canvas canvas = new Canvas(outputBitmap);
        Paint rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(3);
        rectPaint.setColor(Color.YELLOW);
        Paint textPaint = new Paint();
        textPaint.setTextSize(20);
        textPaint.setColor(Color.BLACK);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setStrokeWidth(0);
        Paint textRectPaint = new Paint();
        textRectPaint.setColor(Color.RED);
        textRectPaint.setStyle(Paint.Style.FILL);
        for (YoloResult result : output)
        {
            Rect rect = result.getRect(imageBitmap.getWidth(), imageBitmap.getHeight());
            Rect textRect = new Rect(rect.left, rect.top, rect.left + 150,rect.top + 40);
            canvas.drawRect(rect, rectPaint);
            canvas.drawRect(textRect, textRectPaint);
            canvas.drawText(CocoClasses.get(result.getCls()), textRect.left + 10, textRect.top + 20, textPaint);
        }

        return outputBitmap;
    }
}
