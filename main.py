from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, row_number
from pyspark.sql.window import Window
import argparse
import os

DEFAULT_DATA_DIR = "./data"
DEFAULT_RESULT_DIR = "./"


def main(customers_dir: str, products_dir: str, orders_dir: str, result_dir: str) -> None:
    spark = SparkSession.builder.appName("PopularProductAnalysis").getOrCreate()

    customer_df = spark.read.option("delimiter", "\t").csv(customers_dir, header=False, inferSchema=True) \
                        .toDF("customerID", "customer_name", "email", "joinDate", "status")
    product_df = spark.read.option("delimiter", "\t").csv(products_dir, header=False, inferSchema=True) \
                        .toDF("productID", "product_name", "price", "numberOfProducts")
    order_df = spark.read.option("delimiter", "\t").csv(orders_dir, header=False, inferSchema=True) \
                        .toDF("customerID", "orderID", "productID", "numberOfProduct", "orderDate", "status")
    # print(customer_df.show(), product_df.show(), order_df.show())

    # filter only relevant values
    active_customer_df = customer_df.filter(col("status") == "active")
    delivered_order_df = order_df.filter(col("status") == "delivered")

    joined_df = delivered_order_df.join(active_customer_df, "customerID", "inner") \
                                .join(product_df, "productID", "inner")

    window_spec = Window.partitionBy("customerID").orderBy(desc("numberOfProduct"))
    ranked_df = joined_df.withColumn("rank", row_number().over(window_spec))

    most_popular_products = ranked_df.filter(col("rank") == 1) \
                                    .select("customer_name", "product_name")

    result_path = os.path.join(result_dir, 'result.csv')
    most_popular_products.write.mode("overwrite").csv(result_path, header=True)

    spark.stop()


if __name__ == "__main__":
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
    main(customer_path, product_path, order_path, args.result_dir)
    