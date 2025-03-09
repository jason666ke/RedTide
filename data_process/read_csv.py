import argparse
import pandas as pd

def preview_csv(file_path, rows=5):
    """
    预览CSV文件的前N行数据
    
    Args:
        file_path: CSV文件路径
        rows: 需要预览的行数，默认为5
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 打印基本信息
        print("\n文件基本信息:")
        print("-" * 50)
        print(f"总行数: {len(df)}")
        print(f"总列数: {len(df.columns)}")
        print(f"列名: {', '.join(df.columns)}")
        
        # 打印数据预览
        print("\n数据预览:")
        print("-" * 50)
        print(df.head(rows))
        
        # 打印数据类型信息
        print("\n数据类型信息:")
        print("-" * 50)
        print(df.dtypes)
        
    except Exception as e:
        print(f"错误: 无法读取文件 - {str(e)}")

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='预览CSV文件内容')
    parser.add_argument('file_path', type=str, help='CSV文件路径')
    parser.add_argument('--rows', type=int, default=5, help='需要预览的行数(默认: 5)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 预览文件
    preview_csv(args.file_path, args.rows)

if __name__ == '__main__':
    main()