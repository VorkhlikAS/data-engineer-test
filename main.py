from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, dense_rank
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType
import argparse
import os

DEFAULT_DATA_DIR = "./data"
DEFAULT_RESULT_DIR = "./"


def main() -> None:
    parser = argparse.ArgumentParser(description="Popular Products with PySpark")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing customers, products and orders files", default=DEFAULT_DATA_DIR)
    parser.add_argument("--result_dir", type=str, help="Path to the result directory", default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist.")
        exit(1)
    elif not os.path.exists(args.result_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist.")
        exit(1)

    # Update file paths
    customer_path = os.path.join(args.data_dir, "Customer.csv")
    product_path = os.path.join(args.data_dir, "Product.csv")
    order_path = os.path.join(args.data_dir, "Order.csv")

    # Check if the specified files exist
    for file_path in [customer_path, product_path, order_path]:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            exit(1)
    process(customer_path, product_path, order_path, args.result_dir)


def process(customers_dir: str, products_dir: str, orders_dir: str, result_dir: str) -> None:
    customer_schema = StructType([
        StructField("customerID", IntegerType(), True),
        StructField("customer_name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("joinDate", DateType(), True),
        StructField("status", StringType(), True)
    ])

    product_schema = StructType([
        StructField("productID", IntegerType(), True),
        StructField("product_name", StringType(), True),
        StructField("price", IntegerType(), True),
        StructField("numberOfProductsRemaining", IntegerType(), True)
    ])

    order_schema = StructType([
        StructField("customerID", IntegerType(), True),
        StructField("orderID", IntegerType(), True),
        StructField("productID", IntegerType(), True),
        StructField("numberOfProduct", IntegerType(), True),
        StructField("orderDate", DateType(), True),
        StructField("status", StringType(), True)
    ])
    
    spark = SparkSession.builder.appName("PopularProductAnalysis").getOrCreate()

    try: 
        customer_df = spark.read.option("delimiter", "\t").csv(customers_dir, header=False, schema=customer_schema)
        product_df = spark.read.option("delimiter", "\t").csv(products_dir, header=False, schema=product_schema)
        order_df = spark.read.option("delimiter", "\t").csv(orders_dir, header=False, schema=order_schema)
    except Exception as e:
        print(f"Error: incorrect file structure\n{e}")
        spark.stop()
        exit(1)

    # filter only relevant values
    active_customer_df = customer_df.filter(col("status") == "active")
    delivered_order_df = order_df.filter(col("status") == "delivered")

    joined_df = active_customer_df.join(delivered_order_df, "customerID", "inner") \
        .join(product_df, "ProductID", "inner")
    
    joined_df = joined_df.withColumn("total_cost", joined_df.price * joined_df.numberOfProduct)
    # joined_df = joined_df.withColumn("total_cost", col("total_cost").cast(IntegerType()))
    
    
    popular_products = joined_df.groupBy("customerID", "productID", "customer_name", "product_name") \
        .sum("total_cost") \
        .withColumnRenamed("sum(total_cost)", "total_cost")
    
    # print(popular_products.show())
    windowSpec = Window.partitionBy("customerID").orderBy(desc("total_cost"))
    result_df = popular_products.withColumn("dense_rank", dense_rank().over(windowSpec)) \
                                    .filter(col("dense_rank") == 1) \
                                    .select("customer_name", "product_name")
    # print(result_df.show())

    result_path = os.path.join(result_dir, 'result.csv')
    
    result_df.write.mode("overwrite").csv(result_path, header=True)

    spark.stop()


if __name__ == "__main__":
    main()
    