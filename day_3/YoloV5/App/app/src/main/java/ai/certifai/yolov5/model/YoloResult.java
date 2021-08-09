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

import android.graphics.Rect;

/**
 * Result predicted by YOLOv5
 *
 * @author YCCertifai
 */
public class YoloResult
{
    private float x;
    private float y;
    private float w;
    private float h;
    private float conf;
    private int cls;
    private static int imageWidth;
    private static int imageHeight;

    public YoloResult(float x, float y, float w, float h, float conf, int cls)
    {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.conf = conf;
        this.cls = cls;
    }

    public static void setImageSize(int[] imageSize)
    {
        YoloResult.imageWidth = imageSize[0];
        YoloResult.imageHeight = imageSize[1];
    }

    @Override
    public String toString()
    {
        return String.format("x: %f, y: %f, w: %f, h: %f, conf: %f, cls: %d", x, y, w, h, conf, cls);
    }

    public Rect getRect()
    {
        int left = (int) getLeft();
        int right = (int) getRight();
        int top = (int) getTop();
        int bot = (int) getBot();
        return new Rect(left, top, right, bot);
    }

    public Rect getRect(int width, int height)
    {
        float ratio = Math.max((float) width / imageWidth, (float) height / imageHeight);

        int left = (int) (getLeft() * ratio - ((imageWidth * ratio - width) / 2));
        int right = (int) (getRight() * ratio - ((imageWidth * ratio - width) / 2));
        int top = (int) (getTop() * ratio - ((imageWidth * ratio - height) / 2)) ;
        int bot = (int) (getBot() * ratio - ((imageWidth * ratio - height) / 2));
        return new Rect(left, top, right, bot);
    }

    public float getArea()
    {
        return w * h;
    }

    public float getLeft()
    {
        return x - w / 2;
    }

    public float getRight()
    {
        return x + w / 2;
    }

    public float getTop()
    {
        return y - h / 2;
    }

    public float getBot()
    {
        return y + h / 2;
    }

    public float getConf()
    {
        return conf;
    }

    public float getX()
    {
        return x;
    }

    public float getY()
    {
        return y;
    }

    public float getW()
    {
        return w;
    }

    public float getH()
    {
        return h;
    }

    public int getCls()
    {
        return cls;
    }
}