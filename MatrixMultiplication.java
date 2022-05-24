import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Random;

//每个map或reduce任务生成一个MatrixMultiplication实例
//static每次都为初始值

/**
 * @author joe
 * @since 2021.6.28
 * @version 3.2
 */
public class MatrixMultiplication {
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException, URISyntaxException {
		// 初始化输入数据
		init();

		// 配置类 可addResource加入xml配置文件
		Configuration conf = new Configuration();

		// 输入输出路径
		Path input1 = new Path("/MatrixMul/input/MatrixA");
		Path input2 = new Path("/MatrixMul/input/MatrixB");
		Path output = new Path("/MatrixMul/output");

		// 初始化job
		Job job = Job.getInstance(conf, "Matrix Multiplication");

		// 设置主类
		job.setJarByClass(MatrixMultiplication.class);

		// 设置Reduce输出
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(LongWritable.class);
		// 设置Map输出
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);

		// 设置Map任务的类
		job.setMapperClass(MatrixMapper.class);
		// 设置Reduce任务的类
		job.setReducerClass(MatrixReducer.class);

		// 设置输入方式
		job.setInputFormatClass(TextInputFormat.class);
		// 设置输出方式
		job.setOutputFormatClass(TextOutputFormat.class);
		// 设置输入的job和路径
		FileInputFormat.setInputPaths(job, input1, input2);
		// 删除已经存在的输出文件夹
		output.getFileSystem(conf).delete(output, true);
		// 设置输出文件夹
		FileOutputFormat.setOutputPath(job, output);

		// 等待作业完成
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

	/**
	 * 造数据到本地
	 * 将本地文件上传到HDFS
	 * @throws IOException 输出到文件可能产生异常
	 * @throws URISyntaxException 路径可能不存在
	 */
	private static void init() throws IOException, URISyntaxException {
		// 随机化
		Random random = new Random();
		OutputStreamWriter writer;
		String input1 = "/home/joe/文档/Big Data/CourseDesign/MatrixMultiplication/Data/MatrixA";
		String input2 = "/home/joe/文档/Big Data/CourseDesign/MatrixMultiplication/Data/MatrixB";

		// 写入矩阵A
		writer = new OutputStreamWriter(new FileOutputStream(input1), StandardCharsets.UTF_8);
		for (int i = 1; i <= MatSize.m; i++) {
			for (int j = 1; j <= MatSize.l; j++) {
				writer.write(random.nextInt() + (j == MatSize.l ? "\n" : ","));
			}
		}
		writer.close();

		// 写入矩阵B
		writer = new OutputStreamWriter(new FileOutputStream(input2), StandardCharsets.UTF_8);
		for (int i = 1; i <= MatSize.l; i++) {
			for (int j = 1; j <= MatSize.n; j++) {
				writer.write(random.nextInt() + (j == MatSize.n ? "\n" : ","));
			}
		}
		writer.close();

		// 将本地文件复制到HDFS相应目录
		FileSystem fs = FileSystem.get(new URI("hdfs://localhost:9000"), new Configuration());
		fs.copyFromLocalFile(new Path(input1), new Path("/MatrixMul/input"));
		fs.copyFromLocalFile(new Path(input2), new Path("/MatrixMul/input"));
		fs.close();
	}

	/**
	 * 执行Map任务的类
	 * 继承Mapper
	 */
	public static class MatrixMapper extends Mapper<LongWritable, Text, Text, Text> {
		private String matrixName = null;
		private int rowIndexA = 1;
		private int rowIndexB = 1;

		/**
		 * 获得矩阵名称
		 * 
		 * @param context 传递数据的上下文
		 */
		@Override
		protected void setup(Context context) {
			matrixName = ((FileSplit) context.getInputSplit()).getPath().getName();
		}

		/**
		 * 重写map方法
		 * 
		 * @param key 行号
		 * @param value 一行的数据
		 * @param context 传递数据的上下文
		 * @throws IOException write可能出现异常
		 * @throws InterruptedException 某些对象可能不存在
		 */
		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			// 获取一行元素
			String[] tokens = value.toString().split(",");
			if ("MatrixA".equals(matrixName)) {
				// 遍历每个元素
				for (int j = 0; j < tokens.length; j++) {
					// 忽略0元素
					if ("0".equals(tokens[j])) {
						continue;
					}
					// 枚举C的列
					// A[row][j]都可用于计算C[row][i]
					for (int i = 1; i <= MatSize.n; i++) {
						context.write(new Text(rowIndexA + "," + i), new Text("A," + (j + 1) + "," + tokens[j]));
					}
				}
				++rowIndexA;
			} else if ("MatrixB".equals(matrixName)) {
				// 遍历每个元素
				for (int j = 0; j < tokens.length; j++) {
					// 忽略0元素
					if ("0".equals(tokens[j])) {
						continue;
					}
					// 枚举C的行
					// B[row][j]都可用于计算C[i][j]
					for (int i = 1; i <= MatSize.m; i++) {
						context.write(new Text(i + "," + (j + 1)), new Text("B," + rowIndexB + "," + tokens[j]));
					}
				}
				++rowIndexB;
			}
		}
	}

	/**
	 * 执行Reduce任务的类
	 * 继承Reducer
	 */
	public static class MatrixReducer extends Reducer<Text, Text, Text, LongWritable> {
		/**
		 * 重写reduce方法
		 * 
		 * @param key 表示计算的元素C[i][j]
		 * @param values 用于计算C[i][j]的所有值 
		 * @param context 传递数据的上下文
		 * @throws IOException write可能出现异常
		 * @throws InterruptedException 某些对象可能不存在
		 */
		@Override
		protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			/*
				C[i][j]=sum(A[i][k]*B[k][j])
				mapA[k]=A[i][k]
				mapB[k]=B[k][j] 
			*/
			HashMap<String, String> mapA = new HashMap<>();
			HashMap<String, String> mapB = new HashMap<>();
			for (Text value : values) {
				String[] val = value.toString().split(",");
				if ("A".equals(val[0])) {
					mapA.put(val[1], val[2]);
				} else if ("B".equals(val[0])) {
					mapB.put(val[1], val[2]);
				}
			}
			long res = 0;
			// 相应元素相乘再累加
			for (String mkey : mapA.keySet()) {
				if (mapB.get(mkey) == null) {
					continue;
				}
				res += Long.parseLong(mapA.get(mkey)) * Long.parseLong(mapB.get(mkey));
			}
			key.set("(" + key + ")");
			context.write(key, new LongWritable(res));
		}
	}
}