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

package ai.certifai.yolov5.utils;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Class handling asset files
 *
 * @author YCCertifai
 */
public class AssetHandler
{
    public static String assetFilePath(Context context, String assetName) throws IOException
    {
        File file = new File(context.getFilesDir(), assetName);

        if (file.exists() && file.length() > 0)
        {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName);
             OutputStream os = new FileOutputStream(file))
        {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1)
            {
                os.write(buffer, 0, read);
            }
            os.flush();
            return file.getAbsolutePath();
        }
    }
}
