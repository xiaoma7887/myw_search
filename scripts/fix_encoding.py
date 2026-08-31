#!/usr/bin/env python3
"""
修复Windows终端Unicode编码问题
"""

import sys
import io
import os

def add_unicode_support_to_file(filepath):
    """为Python文件添加Unicode支持"""
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经包含编码声明
        if '# -*- coding: utf-8 -*-' not in content:
            # 在shebang行后添加编码声明
            lines = content.split('\n')
            if lines and lines[0].startswith('#!/'):
                lines.insert(1, '# -*- coding: utf-8 -*-')
                lines.insert(2, '')
            else:
                lines.insert(0, '# -*- coding: utf-8 -*-')
                lines.insert(1, '')
            
            # 添加Unicode支持代码
            unicode_support = '''
import sys
import io

# 设置Unicode支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # 对于旧版本Python
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
'''
            
            # 找到第一个导入语句后插入
            import_found = False
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    lines.insert(i, unicode_support)
                    import_found = True
                    break
            
            if not import_found:
                # 如果没有导入语句，在编码声明后插入
                for i, line in enumerate(lines):
                    if '# -*- coding: utf-8 -*-' in line:
                        lines.insert(i + 2, unicode_support)
                        break
            
            new_content = '\n'.join(lines)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✓ 已修复: {filepath}")
            return True
        else:
            print(f"✓ 已包含编码支持: {filepath}")
            return False
            
    except Exception as e:
        print(f"✗ 修复失败 {filepath}: {e}")
        return False

def main():
    """主函数"""
    print("修复Python脚本Unicode编码支持")
    print("=" * 50)
    
    # 需要修复的脚本文件
    scripts_to_fix = [
        "scripts/data_collection/kaggle_downloader.py",
        "scripts/data_collection/taobao_crawler.py", 
        "scripts/data_collection/dataset_builder.py",
        "scripts/data_collection/web_crawler.py",
        "scripts/data_collection/test_taobao_api.py",
        "scripts/get_data.py",
        "scripts/setup_database.py",
        "scripts/train_model.py",
        "main.py"
    ]
    
    fixed_count = 0
    
    for script_path in scripts_to_fix:
        if os.path.exists(script_path):
            if add_unicode_support_to_file(script_path):
                fixed_count += 1
        else:
            print(f"- 文件不存在: {script_path}")
    
    print(f"\n修复完成! 共修复 {fixed_count} 个文件")
    print("\n现在脚本应该可以正常显示Unicode表情符号了!")

if __name__ == "__main__":
    main()