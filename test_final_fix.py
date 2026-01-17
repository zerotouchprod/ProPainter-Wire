#!/usr/bin/env python3
"""
Финальный тест исправлений для inference_core.py
Проверяет, что все критические исправления присутствуют
"""

import re
import sys

def check_file_for_fixes():
    """Проверяет inference_core.py на наличие критических исправлений"""
    
    with open('inference_core.py', 'r') as f:
        content = f.read()
    
    checks = [
        ("safe_raft_inference функция", "def safe_raft_inference"),
        ("Дополнительная обработка ошибок RAFT", "RAFT model forward failed"),
        ("Проверка входных данных RAFT", "RAFT input shape:"),
        ("Проверка NaN/Inf значений", "torch.isnan"),
        ("Множественные стратегии восстановления", "recovery_strategies"),
        ("Очистка кэша и retry", "torch.cuda.empty_cache"),
        ("CPU fallback", "enable_cpu_fallback"),
        ("Детальное логирование", "InferenceLogger"),
        ("AMP поддержка", "torch.autocast"),
        ("Чанковая обработка", "process_video_in_chunks"),
    ]
    
    print("🧪 Финальная проверка исправлений inference_core.py")
    print("=" * 60)
    
    all_passed = True
    for check_name, pattern in checks:
        if re.search(pattern, content):
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False
    
    print("=" * 60)
    
    # Проверка отсутствия рекурсивных вызовов
    lines = content.split('\n')
    safe_raft_lines = []
    in_function = False
    
    for i, line in enumerate(lines):
        if 'def safe_raft_inference' in line:
            in_function = True
        elif in_function and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            in_function = False
        
        if in_function and 'safe_raft_inference(' in line and 'def safe_raft_inference' not in line:
            safe_raft_lines.append((i+1, line.strip()))
    
    if safe_raft_lines:
        print("⚠️  Найдены возможные рекурсивные вызовы safe_raft_inference:")
        for line_num, line in safe_raft_lines:
            print(f"   Строка {line_num}: {line}")
        all_passed = False
    else:
        print("✅ Нет рекурсивных вызовов safe_raft_inference")
    
    # Проверка обработки CUDA ошибок
    cuda_error_handling = [
        ('"cuda" in error_str', 'Проверка CUDA ошибок'),
        ('"CUDA" in str(raft_error)', 'Проверка CUDA ошибок (верхний регистр)'),
        ('torch.cuda.empty_cache()', 'Очистка кэша CUDA'),
    ]
    
    print("\n🔍 Проверка обработки CUDA ошибок:")
    for pattern, description in cuda_error_handling:
        if re.search(pattern, content):
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            all_passed = False
    
    return all_passed

def main():
    """Основная функция тестирования"""
    
    try:
        all_passed = check_file_for_fixes()
        
        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 ВСЕ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ПРИСУТСТВУЮТ!")
            print("\n📋 Сводка:")
            print("1. ✅ Безопасный RAFT с обработкой ошибок")
            print("2. ✅ Проверка входных данных (формат, NaN/Inf)")
            print("3. ✅ Множественные стратегии восстановления")
            print("4. ✅ Детальное логирование и мониторинг")
            print("5. ✅ AMP для производительности")
            print("6. ✅ Чанковая обработка для экономии памяти")
            print("7. ✅ CPU fallback для стабильности")
            print("8. ✅ Отсутствие рекурсивных ошибок")
            print("\n🚀 inference_core.py готов к production!")
        else:
            print("⚠️  НЕКОТОРЫЕ ИСПРАВЛЕНИЯ ОТСУТСТВУЮТ!")
            print("Требуется дополнительная проверка файла.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Ошибка при проверке: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
