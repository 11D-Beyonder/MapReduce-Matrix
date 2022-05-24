import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
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
import java.util.Random;

/**
 * @author 11D_Beyonder
 * @version 1.3
 * @since 2021.8.20
 */
public class MatrixModification {
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException, URISyntaxException {
		// 初始化
		init();

		// 任务配置
		Configuration conf = new Configuration();

		// 输入输出路径
		Path input = new Path("/MatrixMul/input/InitialMatrix");
		Path output = new Path("/MatrixMul/output");

		// 初始化job
		Job job = Job.getInstance(conf, "Matrix Modification");

		// 设置主类
		job.setJarByClass(MatrixModification.class);

		// 设置Reduce输出
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(LongWritable.class);
		// 设置Map输出
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);

		// 设置Map任务的类
		job.setMapperClass(MatrixModification.PrepareMapper.class);
		// 设置Reduce任务的类
		job.setReducerClass(MatrixModification.PrepareReducer.class);

		// 设置输入输出方式
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		// 设置输入路径
		FileInputFormat.setInputPaths(job, input);
		// 删除原有的输出目录
		output.getFileSystem(conf).delete(output, true);
		// 设置输出路径
		FileOutputFormat.setOutputPath(job, output);

		// 等待任务结束
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

	/**
	 * 执行Map任务的类 
	 * 继承Mapper
	 */
	public static class PrepareMapper extends Mapper<LongWritable, Text, Text, Text> {
		// 行索引
		private int rowIndex = 1;
		// 8个方向
		final int[][] dirStep = {{-1, -1}, {1, 1}, {-1, 1}, {1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};

		/**
		 * 重写map方法
		 * 
		 * @param key     行号
		 * @param value   一行的数据
		 * @param context 传递数据的上下文
		 * @throws IOException          write可能出现异常
		 * @throws InterruptedException 某些对象可能不存在
		 */
		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			// 获取一行数据
			String[] tokens = value.toString().split(",");
			// 枚举元素
			for (int j = 0; j < tokens.length; j++) {
				int dirRow = -1;
				int dirColumn = -1;
				// 处理斜方向的位置
				for (int v = 0; v < 4; v++) {
					dirRow = rowIndex + dirStep[v][0];
					dirColumn = j + 1 + dirStep[v][1];
					// 在矩阵范围内就添加权值为1的值
					if (1 <= dirRow && dirRow <= MatSize.m && 1 <= dirColumn && dirColumn <= MatSize.n) {
						context.write(new Text(dirRow + "," + dirColumn), new Text("1," + tokens[j]));
					}
				}
				// 处理其余位置
				for (int v = 4; v < 8; v++) {
					dirRow = rowIndex + dirStep[v][0];
					dirColumn = j + 1 + dirStep[v][1];
					// 在矩阵范围内就添加权值为2的值
					if (1 <= dirRow && dirRow <= MatSize.m && 1 <= dirColumn && dirColumn <= MatSize.n) {
						context.write(new Text(dirRow + "," + dirColumn), new Text("2," + tokens[j]));
					}
				}
				context.write(new Text(rowIndex + "," + (j + 1)), new Text("-1," + tokens[j]));
			}
			++rowIndex;
		}
	}

	/**
	 * 执行Reduce任务的类 
	 * 继承Reducer
	 */
	public static class PrepareReducer extends Reducer<Text, Text, Text, LongWritable> {
		/**
		 * 重写reduce方法
		 * 
		 * @param key     表示计算的元素C[i][j]
		 * @param values  用于计算C[i][j]的所有值
		 * @param context 传递数据的上下文
		 * @throws IOException          write可能出现异常
		 * @throws InterruptedException 某些对象可能不存在
		 */
		@Override
		protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			long tot = 0;
			long weight = 0;
			long initialValue = 0;
			for (Text value : values) {
				String[] val = value.toString().split(",");
				// -1 表示初始值
				// 1、2表示权值
				if ("-1".equals(val[0])) {
					initialValue = Long.parseLong(val[1]);
				} else {
					weight += Long.parseLong(val[0]);
					tot += Long.parseLong(val[1]) * Long.parseLong(val[0]);
				}
			}
			double worthValue = (double) tot / (double) weight;
			// 误差估计
			if (Math.abs(worthValue - initialValue) / (double) Math.abs(initialValue) >= 0.3) {
				context.write(new Text("(" + key.toString() + ")"), new LongWritable(Math.round(worthValue)));
			} else {
				context.write(new Text("(" + key.toString() + ")"), new LongWritable(initialValue));
			}
		}
	}

	/**
	 * 造数据到本地
	 * 将本地文件上传到HDFS
	 * @throws IOException 输出到文件可能产生异常
	 * @throws URISyntaxException 路径可能不存在
	 */
	private static void init() throws IOException, URISyntaxException {
		Random random = new Random();
		OutputStreamWriter writer;
		String input = "/home/joe/文档/Big Data/CourseDesign/MatrixMultiplication/Data/InitialMatrix";

		// 随机造数
		writer = new OutputStreamWriter(new FileOutputStream(input), StandardCharsets.UTF_8);
		for (int i = 1; i <= MatSize.m; i++) {
			for (int j = 1; j <= MatSize.n; j++) {
				writer.write(random.nextInt() + (j == MatSize.n ? "\n" : ","));
			}
		}

		writer.close();

		// 复制文件到HDFS
		FileSystem fs = FileSystem.get(new URI("hdfs://localhost:9000"), new Configuration());
		fs.copyFromLocalFile(new Path(input), new Path("/MatrixMul/input"));
		fs.close();
	}
}
