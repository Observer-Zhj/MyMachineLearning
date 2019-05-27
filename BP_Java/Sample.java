package cn.zhj.deeplearning;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Created by admin on 2018/10/19.
 * 随机不重复抽样。
 * 随机从DataSet中抽取一部分输入数据集和相对应的输出数据集，输入数据集和输出数据集都保存在二维数组中，
 */
public class Sample {

    private int[] index;
    private boolean isShuffling = false;
    private int end = 0;
    private int len = 0;
    private DataSet dataSet;

    public Sample(DataSet dataSet) {
        this.dataSet = dataSet;
        len = this.dataSet.getNumSample();
        index = new int[len];
        for (int i = 0; i < len; i++) {
            index[i] = i;
        }
    }

    private static void swap(@NotNull int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    public static void shuffleArray(@NotNull int[] array) {
        Random rnd = new Random();
        for (int i = array.length; i > 1; i--) {
            swap(array, i-1, rnd.nextInt(i));
        }
    }

    /**
     * 打乱数据并生成一个迭代器。
     * 如果是第一次抽样，则会先打乱输入数据集和输出数据集，输出数据集打乱顺序与输入数据集一样，即打乱后两个数据集也一一对应，然后从中抽取一个`batch`的数据。
     * 后面每次调用nextBatch方法都会从中取出一个'batch'的数据。在一轮数据被取完前，取出的数据不会重复。如果取到最后剩余数据量小于`batch`，则全部取出。
     * 下一次调用nextBatch方式时会重新打乱数据。
     * @param batch 要取出的样本数。
     * @return 数组，第一个元素是输入数据集，第二个元素是输出数据集。
     */
    public double[][][] nextBatch(int batch) {
        if (batch > len)
            throw new IllegalArgumentException("batch大于样本数，无法抽样");
        if (!isShuffling) {
            shuffleArray(index);
            for (int i : index) {
                System.out.print(i);
            }
            System.out.println();
            isShuffling = true;
        }
        if (end + batch > len) {
            batch = len - end;
        }

        double[][] inpBatch = new double[batch][];
        double[][] outBatch = new double[batch][];
        for (int i = 0; i < batch; i++) {
            inpBatch[i] = dataSet.getInput()[index[end] + i].clone();
            outBatch[i] = dataSet.getOutput()[index[end] + i].clone();
        }
        ArrayList<double[][]> result = new ArrayList<double[][]>();
        result.add(inpBatch);
        result.add(outBatch);

        end += batch;
        if (end >= len) {
            end = 0;
            isShuffling = false;
        }
        double[][][] res = {inpBatch, outBatch};
        return res;
    }
}
