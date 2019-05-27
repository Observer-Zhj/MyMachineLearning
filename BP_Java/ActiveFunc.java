package cn.zhj.deeplearning;

import org.jetbrains.annotations.Contract;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by admin on 2018/10/18.
 */
public final class ActiveFunc {
    public final static ArrayList<String> actFuncCollection =
            new ArrayList<String>(Arrays.asList(new String[]{null, "sigmoid", "tanh", "relu"}));

    /** sigmoid函数，1/(1+e^(-x)) */
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /** sigmoid函数的导函数，e^(-x)/(1+e^(-x))^2或sigmoid(x)*(1-sigmoid(x)) */
    public static double diffSigmoid(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    /** 双曲正切函数tanh，(e^x-e^(-x))/(e^x+e^(-x)) */
    public static double tanh(double x) {
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
    }

    /** 双曲正切函数tanh的导函数，1-tanh(x)*tanh(x) */
    public static double diffTanh(double x) {
        return 1 - tanh(x) * tanh(x);
    }

    /** 线性整流函数relu，if x > 0 x else 0 */
    @Contract(pure = true)
    public static double relu(double x) {
        return Math.max(0, x);
    }

    /** 线性整流函数relu的导函数，if x > 0 1 else 0 */
    @Contract(pure = true)
    public static double diffRelu(double x) {
        if (x > 0)
            return 1.0;
        else
            return 0.0;
    }

    /** 计算激活值 */
    public static double active(double x, String actFunc) {
        if (actFunc == null) {
            return x;
        } else if (actFunc.equals("sigmoid")) {
            return sigmoid(x);
        } else if (actFunc.equals("relu")) {
            return relu(x);
        } else if (actFunc.equals("tanh")) {
            return tanh(x);
        } else {
            throw new IllegalArgumentException("actFunc不是有效的参数");
        }
    }

    public static double diffActive(double x, String actFunc) {
        if (actFunc == null) {
            return 0.0;
        } else if (actFunc.equals("sigmoid")) {
            return diffSigmoid(x);
        } else if (actFunc.equals("relu")) {
            return diffRelu(x);
        } else if (actFunc.equals("tanh")) {
            return diffTanh(x);
        } else {
            throw new IllegalArgumentException("actFunc不是有效的参数");
        }
    }
}
