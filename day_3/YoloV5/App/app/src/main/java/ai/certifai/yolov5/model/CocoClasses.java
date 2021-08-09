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

/**
 * Enum containing COCO classes
 *
 * @author YCCertifai
 */
public enum CocoClasses
{
    PERSON,
    BICYCLE,
    CAR,
    MOTORCYCLE,
    AIRPLANE,
    BUS,
    TRAIN,
    TRUCK,
    BOAT,
    TRAFFIC_LIGHT,
    HYDRANT,
    STOP_SIGN,
    PARKING_METER,
    BENCH,
    BIRD,
    CAT,
    DOG,
    HORSE,
    SHEEP,
    COW,
    ELEPHANT,
    BEAR,
    ZEBRA,
    GIRAFFE,
    BACKPACK,
    UMBRELLA,
    HANDBAG,
    TIE,
    SUITCASE,
    FRISBEE,
    SKIS,
    SNOWBOARD,
    SPORTS_BALL,
    KITE,
    BASEBALL_BAT,
    BASEBALL_GLOVE,
    SKATEBOARD,
    SURFBOARD,
    TENNIS_RACKET,
    BOTTLE,
    WINE_GLASS,
    CUP,
    FORK,
    KNIFE,
    SPOON,
    BOWL,
    BANANA,
    APPLE,
    SANDWICH,
    ORANGE,
    BROCCOLI,
    CARROT,
    HOT_DOG,
    PIZZA,
    DONUT,
    CAKE,
    CHAIR,
    COUCH,
    POTTED_PLANT,
    BED,
    DINING_TABLE,
    TOILET,
    TV,
    LAPTOP,
    MOUSE,
    REMOTE,
    KEYBOARD,
    CELL_PHONE,
    MICROWAVE,
    OVEN,
    TOASTER,
    SINK,
    REFRIGERATOR,
    BOOK,
    CLOCK,
    VASE,
    SCISSORS,
    TEDDY_BEAR,
    HAIR_DRIER,
    TOOTHBRUSH;

    public static String get(int i)
    {
        return CocoClasses.values()[i].toString();
    }

    @Override
    public String toString()
    {
        return this.name().toLowerCase().replace('_', ' ');
    }
}