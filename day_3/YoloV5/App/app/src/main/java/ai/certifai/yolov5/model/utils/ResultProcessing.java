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

import org.pytorch.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import ai.certifai.yolov5.model.YoloResult;

/**
 * Class processing result of YOLOv5
 *
 * @author YCCertifai
 */
public class ResultProcessing
{
    public static List<YoloResult> processResult(Tensor tensor, double confThreshold, double iouThreshold)
    {
        List<YoloResult> results = filterWithConf(tensor, confThreshold);
        return performNMS(results, iouThreshold);
    }

    private static List<YoloResult> performNMS(List<YoloResult> results, double iouThreshold)
    {
        List<YoloResult> output = new ArrayList<>();
        // sort inversely to conf
        Collections.sort(results, ResultProcessing::compareYoloResultByConf);

        boolean[] removed = new boolean[results.size()];
        Arrays.fill(removed, false);
        int removedNum = results.size();

        for (int i = 0; i < results.size() && removedNum > 0; i++)
        {
            if (removed[i]) continue;

            YoloResult result = results.get(i);
            output.add(result);
            removedNum--;

            for (int j = i + 1; j < results.size(); j++)
            {
                YoloResult comparator = results.get(j);

                if (performIOU(result, comparator) > iouThreshold || removed[j])
                {
                    removed[j] = true;
                    removedNum--;
                }
            }
        }
        return output;
    }

    private static float performIOU(YoloResult result1, YoloResult result2)
    {
        float area1 = result1.getArea();
        float area2 = result2.getArea();
        if (area1 <= 0 || area2 <= 0) return 0;

        float intMinX = Math.max(result1.getLeft(), result2.getLeft());
        float intMaxX = Math.min(result1.getRight(), result2.getRight());
        float intMinY = Math.max(result1.getTop(), result2.getTop());
        float intMaxY = Math.min(result1.getBot(), result2.getBot());
        float intArea = Math.max(intMaxX - intMinX, 0) * Math.max(intMaxY - intMinY, 0);

        return intArea / (area1 + area2 - intArea);
    }

    private static int compareYoloResultByConf(YoloResult result1, YoloResult result2)
    {
        return Float.compare(result2.getConf(), result1.getConf());
    }

    private static List<YoloResult> filterWithConf(Tensor tensor, double confThreshold)
    {
        List<YoloResult> results = new ArrayList<>();
        int outputRow = (int) tensor.shape()[1];
        int outputCol = (int) tensor.shape()[2];
        int numCls = outputCol - 5;
        // flatten the tensor
        float[] outputs = tensor.getDataAsFloatArray();

        for (int i = 0; i < outputRow; i++)
        {
            // format of result -> [x, y, w, h, conf, [class]]
            int curRowStartIdx = i * outputCol;
            int curConfIdx = curRowStartIdx + 4;
            float conf = outputs[curConfIdx];
            if (conf > confThreshold)
            {
                float x = outputs[curRowStartIdx];
                float y = outputs[curRowStartIdx + 1];
                float w = outputs[curRowStartIdx + 2];
                float h = outputs[curRowStartIdx + 3];

                int cls = getMaxCls(outputs, curRowStartIdx, numCls);

                results.add(new YoloResult(x, y, w, h, conf, cls));
            }
        }
        return results;
    }

    private static int getMaxCls(float[] outputs, int row, int numCls)
    {
        float max = 0;
        int cls = 0;
        for (int i = 0; i < numCls; i++)
        {
            // format of result -> [x, y, w, h, conf, [class]]
            float conf = outputs[row + 5 + i];
            if (conf > max)
            {
                max = conf;
                cls = i;
            }
        }
        return cls;
    }
}
