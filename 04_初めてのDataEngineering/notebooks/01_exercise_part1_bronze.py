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

# カタログの作成（存在しない場合）
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")

print(f"✅ カタログ {CATALOG} を使用します")

# COMMAND ----------

# スキーマの作成（存在しない場合）
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

print(f"✅ スキーマ {CATALOG}.{SCHEMA} を使用します")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ソースデータの確認
# MAGIC
# MAGIC NYCタクシーのサンプルデータを使用します。

# COMMAND ----------

# サンプルデータの読み込み
source_df = spark.read.table("samples.nyctaxi.trips")

# データの確認
print(f"レコード数: {source_df.count():,}")
source_df.printSchema()

# COMMAND ----------

# 先頭10件を表示
display(source_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Bronze層の作成
# MAGIC
# MAGIC Bronze層では、ソースデータをそのまま取り込みます。
# MAGIC 実際の業務では、ファイルやAPIから取り込んだ生データをここに保存します。

# COMMAND ----------

# Bronzeテーブルの作成
bronze_df = spark.read.table("samples.nyctaxi.trips")

# テーブルとして保存
bronze_df.write.mode("overwrite").saveAsTable("bronze_trips")

print("✅ Bronze層を作成しました")

# COMMAND ----------

# 確認
display(spark.table("bronze_trips").limit(5))
