# Databricks notebook source
# MAGIC %md
# MAGIC # 演習1: PySparkによる命令型ETL（25分）
# MAGIC
# MAGIC このノートブックでは、Part 1で学んだSparkの基礎を使って、命令型のETLパイプラインを構築します。
# MAGIC
# MAGIC ## 学習目標
# MAGIC - PySparkでデータを読み込み、変換し、保存する
# MAGIC - Bronze → Silver → Gold のメダリオンアーキテクチャを実装する
# MAGIC - 命令型アプローチの特徴を理解する

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 環境設定

# COMMAND ----------

# ユーザー名を自動取得
user_name = spark.sql("SELECT current_user()").first()[0]
user_name_prefix = user_name.split("@")[0].replace(".", "_").replace("-", "_")

# カタログとスキーマの設定
CATALOG = f"workspace_{user_name_prefix}"
SCHEMA = f"de_handson_{user_name_prefix}"

print(f"ユーザー: {user_name}")
print(f"カタログ: {CATALOG}")
print(f"スキーマ: {SCHEMA}")

# COMMAND ----------

# カタログ利用（存在しない場合）
spark.sql(f"USE CATALOG {CATALOG}")

print(f"✅ カタログ {CATALOG} を使用します")

# COMMAND ----------

# スキーマ利用
spark.sql(f"USE {CATALOG}.{SCHEMA}")

print(f"✅ スキーマ {CATALOG}.{SCHEMA} を使用します")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Silver層の作成
# MAGIC
# MAGIC Silver層では、データのクレンジング（品質向上）を行います。
# MAGIC
# MAGIC ### クレンジング内容
# MAGIC - 不正なデータの除外（料金が0以下、距離が0以下）
# MAGIC - 必要なカラムの選択
# MAGIC - 日付カラムの追加

# COMMAND ----------

from pyspark.sql.functions import col, to_date

# Bronzeから読み込み
bronze_df = spark.table("bronze_trips")

# クレンジング処理
silver_df = (bronze_df
    # 不正データの除外
    .filter(col("fare_amount") > 0)
    .filter(col("trip_distance") > 0)
    # 必要カラムの選択と日付カラムの追加
    .select(
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "pickup_zip",
        "dropoff_zip",
        to_date("tpep_pickup_datetime").alias("pickup_date")
    )
)

# テーブルとして保存
silver_df.write.mode("overwrite").saveAsTable("silver_trips")

print("✅ Silver層を作成しました")

# COMMAND ----------

# 確認：レコード数の比較
bronze_count = spark.table("bronze_trips").count()
silver_count = spark.table("silver_trips").count()

print(f"Bronze: {bronze_count:,} 件")
print(f"Silver: {silver_count:,} 件")
print(f"除外された件数: {bronze_count - silver_count:,} 件")
