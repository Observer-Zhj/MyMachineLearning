package cn.zhj.deeplearning;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;

/**
 * Created by admin on 2018/10/18.
 */
public class DataSet {

    private double[][] input;
    private double[][] output;
    private double[][] inpMostValue = new double[2][];
    private double[][] outMostValue = new double[2][];
    private int numSample;
    private boolean isInpNormalized = false;
    private boolean isOutNormalized = false;


    /**
     * 创建一个DataSet对象
     * 所有传入数据都经过deep copy后保存在对象中。
     *
     * @param input 输入数据集，矩阵格式，每一行是一个样本输入。
     * @param output 输出数据集，矩阵格式，每一行是一个样本输出。
     */
    public DataSet(@NotNull double[][] input, @NotNull double[][] output) {
        if (input.length != output.length) {
            throw new IllegalArgumentException("输入样本个数与输出不一致");
        }
        numSample = input.length;
        if (!isMatrix(input) || !isMatrix(output)) {
            throw new IllegalArgumentException("input或output不是矩阵");
        }
        this.input = deepCopyMatrix(input);
        this.output = deepCopyMatrix(output);

    }

    public double[][] getInput() {
        return input;
    }

    /**
     * 修改输入数据集
     * 需要注意如果前面已经调用了normalizeInp或normalizeAll函数，此方法会自动把输入数据集归一化。
     *
     * @param input 输入数据集，矩阵格式，每一行是一个样本输入。
     */
    public void setInput(double[][] input) {
        if (!isMatrix(input)) {
            throw new IllegalArgumentException("input不是矩阵");
        }
        this.input = deepCopyMatrix(input);
        if (!isInpNormalized) {
            normalizeInp();
        }
    }

    public double[][] getOutput() {
        return output;
    }

    /**
     * 修改输出数据集
     * 需要注意如果前面已经调用了normalizeOut或normalizeAll函数，此方法会自动把输出数据集归一化。
     *
     * @param output 输出数据集，矩阵格式，每一行是一个样本输出。
     */
    public void setOutput(double[][] output) {
        if (!isMatrix(output)) {
            throw new IllegalArgumentException("output不是矩阵");
        }
        this.output = deepCopyMatrix(output);
        if (!isOutNormalized) {
            normalizeOut();
        }
    }

    public double[][] getInpMostValue() {
        return inpMostValue;
    }

    public double[][] getOutMostValue() {
        return outMostValue;
    }

    public int getNumSample() {
        return numSample;
    }

    @Contract(pure = true)
    public static boolean isMatrix(@NotNull double[][] twoDimArray) {
        if (twoDimArray.length == 0) {
            throw new IllegalArgumentException("这是个空矩阵");
        }
        int l = twoDimArray[0].length;
        for (int i = 1; i < twoDimArray.length;i++) {
            if (twoDimArray[i].length != l)
                return false;
        }
        return true;
    }

    /**
     * 矩阵deep copy
     *
     * @param matrix 矩阵
     * @return 深度拷贝后的矩阵
     */
    public static double[][] deepCopyMatrix(@NotNull double[][] matrix) {
        double[][] tarMatrix = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            tarMatrix[i] = matrix[i].clone();
        }
        return tarMatrix;
    }

    /** 打印矩阵 */
    public static void printMatrix(@NotNull double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                System.out.print(matrix[i][j] + "  ");
            }
            System.out.println();
        }
    }

    /**
     * 计算矩阵的每一列的最值.
     *
     * @param matrix 矩阵
     * @return 一个二维数组mostValue，mostValue[0]保存的是矩阵的每一列的最小值，位置与矩阵对应，
     * mostValue[1]保存的是最大值，位置也对应。
     */
    private static double[][] calculateMinMax(@NotNull double[][] matrix) {
        double[][] mostValue = new double[2][matrix[0].length];
        mostValue[0] = matrix[0].clone();
        mostValue[1] = matrix[0].clone();
        for (int i = 1; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] < mostValue[0][j])
                    mostValue[0][j] = matrix[i][j];
                if (matrix[i][j] > mostValue[1][j])
                    mostValue[1][j] = matrix[i][j];
            }
        }
        return mostValue;
    }

    /**
     * 矩阵按列归一化。
     * 设matrix是要归一化的矩阵，jColMin，jColMax是matrix第j列的最小值，最大值，resMatrix是归一化后的矩阵，则
     * resMatrix[i][j] = (matrix[i][j] - jColMin) / (jColMax - jColMin)
     * @param matrix 矩阵。
     * @return 返回归一化后的矩阵。
     */
    public static double[][] normalize(double[][] matrix) {
        double[][] mostValue = calculateMinMax(matrix);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = (matrix[i][j] - mostValue[0][j]) / (mostValue[1][j] - mostValue[0][j]);
            }
        }
        return mostValue;
    }

    /**
     * 反归一化
     * 设matrix是要反归一化的矩阵，mostValue[0][j]，mostValue[1][j]是matrix第j列的最小值/最大值，rawMatrix是归一化后的矩阵，则
     * rawMatrix[i][j] = matrix[i][j] * (mostValue[1][j] - mostValue[0][j]) + mostValue[0][j]
     *
     * @param matrix 要反归一化的矩阵。
     * @param mostValue matrix的极值矩阵，第一行是最小值，第二行是最大值。
     * @return 反归一化后的矩阵。
     */
    public static double[][] antiNormalize(@NotNull double[][] matrix, double[][] mostValue) {
        double[][] rawMatrix = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                rawMatrix[i][j] = matrix[i][j] * (mostValue[1][j] - mostValue[0][j]) + mostValue[0][j];
            }
        }
        return rawMatrix;
    }

    /** 对输入数据集进行归一化 */
    public void normalizeInp() {
        inpMostValue = normalize(input);
        isInpNormalized = true;
    }

    /** 对输出数据集进行归一化 */
    public void normalizeOut() {
        outMostValue = normalize(output);
        isOutNormalized = true;
    }

    /** 对输入输出数据集进行归一化 */
    public void normalizeAll() {
        normalizeInp();
        normalizeOut();
    }


    public static void main(String[] args) {
        double[][] da = {{1, 2, 3},{4, 5, 6},{0, 1, 10}, {8, 9, 3}};
        printMatrix(da);
        DataSet ds = new DataSet(da, da);
        Sample sample = new Sample(ds);
        for (int i = 0; i < 4; i++) {
            printMatrix(sample.nextBatch(1)[0]);
        }


    }
}
