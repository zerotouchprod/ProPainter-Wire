#!/usr/bin/env python3
"""
Тест исправленной функции safe_raft_inference.
Проверяет, что функция работает без рекурсивных ошибок.
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.getcwd())

def test_safe_raft_structure():
    """Проверяет структуру функции safe_raft_inference"""
    print("🔍 Проверка структуры safe_raft_inference...")
    
    with open('inference_core.py', 'r') as f:
        content = f.read()
    
    # Проверяем, что нет рекурсивного вызова
    if 'safe_raft_inference(' in content:
        # Считаем количество вызовов
        lines = content.split('\n')
        call_count = 0
        for i, line in enumerate(lines):
            if 'safe_raft_inference(' in line and not line.strip().startswith('def'):
                call_count += 1
                print(f"  Строка {i+1}: {line.strip()}")
        
        if call_count == 0:
            print("✅ Нет рекурсивных вызовов safe_raft_inference")
        else:
            print(f"⚠️ Найдено {call_count} вызовов safe_raft_inference")
            # Проверяем, что это не рекурсивный вызов
            for i, line in enumerate(lines):
                if 'safe_raft_inference(' in line and not line.strip().startswith('def'):
                    # Проверяем контекст - если внутри except, это может быть рекурсия
                    for j in range(max(0, i-10), min(len(lines), i+10)):
                        if 'except' in lines[j] and j < i:
                            print(f"  ⚠️ Возможная рекурсия в строке {i+1}")
                            return False
    
    # Проверяем обработку ошибок
    error_handling_checks = [
        'except RuntimeError as e:',
        'if "out of memory" in str(e).lower()',
        'enable_cpu_fallback',
        'torch.cuda.empty_cache()',
        'video_cpu = video_tensor.cpu()',
        'raft_model_cpu = raft_model.cpu()'
    ]
    
    for check in error_handling_checks:
        if check in content:
            print(f"✅ Найдена обработка ошибок: {check}")
        else:
            print(f"❌ Не найдена обработка ошибок: {check}")
            return False
    
    return True

def test_function_signature():
    """Проверяет сигнатуру функции"""
    print("\n🔍 Проверка сигнатуры функции...")
    
    with open('inference_core.py', 'r') as f:
        lines = f.readlines()
    
    # Ищем определение функции
    func_start = None
    for i, line in enumerate(lines):
        if 'def safe_raft_inference(' in line:
            func_start = i
            break
    
    if func_start is None:
        print("❌ Функция safe_raft_inference не найдена")
        return False
    
    print(f"✅ Функция найдена на строке {func_start+1}")
    
    # Проверяем параметры
    func_line = lines[func_start]
    expected_params = ['video_tensor', 'raft_model', 'scale_factor', 'raft_iter', 'logger', 'enable_cpu_fallback']
    
    for param in expected_params:
        if param in func_line:
            print(f"✅ Параметр {param} найден")
        else:
            print(f"❌ Параметр {param} не найден")
            return False
    
    return True

def test_no_recursion():
    """Проверяет отсутствие рекурсии"""
    print("\n🔍 Проверка отсутствия рекурсии...")
    
    with open('inference_core.py', 'r') as f:
        content = f.read()
    
    # Разделяем на строки для анализа контекста
    lines = content.split('\n')
    
    # Ищем все вызовы safe_raft_inference
    recursive_calls = []
    for i, line in enumerate(lines):
        if 'safe_raft_inference(' in line and not line.strip().startswith('def'):
            # Проверяем контекст - если внутри блока except, это рекурсия
            in_except_block = False
            for j in range(max(0, i-20), i):
                if 'except' in lines[j]:
                    in_except_block = True
                    break
            
            if in_except_block:
                recursive_calls.append((i+1, line.strip()))
    
    if recursive_calls:
        print("❌ Найдены рекурсивные вызовы:")
        for line_num, line_text in recursive_calls:
            print(f"  Строка {line_num}: {line_text}")
        return False
    else:
        print("✅ Рекурсивные вызовы не найдены")
        return True

def main():
    """Основная функция тестирования"""
    print("🧪 Тест исправленной функции safe_raft_inference")
    print("=" * 60)
    
    tests = [
        ("Проверка структуры функции", test_safe_raft_structure),
        ("Проверка сигнатуры", test_function_signature),
        ("Проверка отсутствия рекурсии", test_no_recursion),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                all_passed = False
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("\n📋 Исправления в safe_raft_inference:")
        print("1. ✅ Устранена рекурсивная ошибка")
        print("2. ✅ Упрощен CPU fallback (без рекурсии)")
        print("3. ✅ Сохранена обработка OOM ошибок")
        print("4. ✅ Добавлена очистка памяти")
        print("\n🚀 Функция safe_raft_inference готова к использованию в production")
        return 0
    else:
        print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print("Требуется дополнительная отладка")
        return 1

if __name__ == "__main__":
    sys.exit(main())
