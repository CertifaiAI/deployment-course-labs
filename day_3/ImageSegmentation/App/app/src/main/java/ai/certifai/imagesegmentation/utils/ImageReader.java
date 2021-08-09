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

package ai.certifai.imagesegmentation.utils;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.ExifInterface;

import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Class handling image reading
 *
 * @author willardsm
 */
public class ImageReader
{
    public static Bitmap readFileToImage(File imageFile)
    {
        String path = imageFile.getAbsolutePath();
        return createBitmap(path);
    }

    private static Bitmap createBitmap(String path)
    {
        int rotation = getRotation(path);

        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);

        Bitmap bitmap = BitmapFactory.decodeFile(path);

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(),
                bitmap.getHeight(), matrix, true);
    }

    private static int getRotation(String path)
    {
        int rotate = 0;
        try
        {
            ExifInterface exif = new ExifInterface(path);
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);

            rotate = orientationToRotation(orientation);
        } catch (IOException ignored) {}

        return rotate;
    }

    private static int orientationToRotation(int orientation)
    {
        switch (orientation)
        {
            case ExifInterface.ORIENTATION_ROTATE_270:
                return 270;
            case ExifInterface.ORIENTATION_ROTATE_180:
                return 180;
            case ExifInterface.ORIENTATION_ROTATE_90:
                return 90;
            default:
                return 0;
        }
    }
}