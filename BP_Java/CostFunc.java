package cn.zhj.deeplearning;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

/**
 * Created by admin on 2018/10/18.
 */
public class CostFunc {

    /**
     * 平方代价函数
     * 设n为样本数，m为每个样本的输出个数，用p表示预测输出，y表示真实输出，
     * 则pij，yij是第i个样本的第j个预测，真实输出，那么第i个样本的代价是
     * cost(i) = 1 / 2 * sum((pij - yij)^2), j = 1, 2,..., m.
     * 总体样本是
     * cost = 1 / n * sum(cost(i)), i = 1, 2,..., n.
     *
     * @param preValue 预测输出
     * @param realValue 真实输出
     * @return 总体代价
     */
    public static double squareCost(double[][] preValue, double[][] realValue) {
        if (!DataSet.isMatrix(preValue)) {
            throw new IllegalArgumentException("preValue不是矩阵");
        }
        if (!DataSet.isMatrix(realValue)) {
            throw new IllegalArgumentException("realValue不是矩阵");
        }
        if (preValue.length != realValue.length || preValue[0].length != realValue[0].length) {
            throw new IllegalArgumentException("preValue与realValue的形状不同");
        }
        double cost = 0;
        for (int i = 0; i < preValue.length; i++) {
            cost += squareCostOneSample(preValue[i], realValue[i]);
        }
        return cost / preValue.length;
    }

    @Contract(pure = true)
    public static double squareCostOneSample(@NotNull double[] preValue, double[] realVaule) {
        double cost = 0;
        for (int i = 0; i < preValue.length; i++) {
            cost += 0.5 * (preValue[i] - realVaule[i]) * (preValue[i] - realVaule[i]);
        }
        return cost;
    }
}
