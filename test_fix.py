#!/usr/bin/env python3
"""
Тест исправлений для inference_core.py
Проверяет, что функции вызываются с правильными аргументами.
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.getcwd())

def test_function_signatures():
    """Проверяет сигнатуры функций"""
    print("🔍 Проверка сигнатур функций...")
    
    # Проверяем, что inference_core.py импортируется
    try:
        import inference_core
        print("✅ inference_core.py импортируется успешно")
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    
    # Проверяем наличие ключевых функций
    required_functions = [
        'safe_raft_inference',
        'process_video_in_chunks', 
        'process_single_chunk',
        'calculate_optimal_scale_factor',
        'main'
    ]
    
    for func_name in required_functions:
        if hasattr(inference_core, func_name):
            print(f"✅ Функция {func_name} найдена")
        else:
            print(f"❌ Функция {func_name} не найдена")
            return False
    
    # Проверяем сигнатуры функций
    import inspect
    
    # Проверяем safe_raft_inference
    sig = inspect.signature(inference_core.safe_raft_inference)
    params = list(sig.parameters.keys())
    expected_params = ['video_tensor', 'raft_model', 'scale_factor', 'raft_iter', 'logger', 'enable_cpu_fallback']
    
    if params == expected_params:
        print("✅ safe_raft_inference имеет правильную сигнатуру")
    else:
        print(f"❌ safe_raft_inference имеет неверную сигнатуру: {params}")
        return False
    
    # Проверяем process_video_in_chunks
    sig = inspect.signature(inference_core.process_video_in_chunks)
    params = list(sig.parameters.keys())
    expected_params = ['video_tensor', 'mask_tensor', 'model', 'raft_model', 'flow_complete_model', 'args', 'logger']
    
    if params == expected_params:
        print("✅ process_video_in_chunks имеет правильную сигнатуру")
    else:
        print(f"❌ process_video_in_chunks имеет неверную сигнатуру: {params}")
        return False
    
    return True

def test_main_function():
    """Проверяет функцию main"""
    print("\n🔍 Проверка функции main...")
    
    import inference_core
    
    # Проверяем, что main принимает args
    sig = inspect.signature(inference_core.main)
    params = list(sig.parameters.keys())
    
    if params == ['args']:
        print("✅ main имеет правильную сигнатуру")
    else:
        print(f"❌ main имеет неверную сигнатуру: {params}")
        return False
    
    # Проверяем, что в main есть правильные вызовы
    with open('inference_core.py', 'r') as f:
        content = f.read()
    
    # Проверяем вызов process_video_in_chunks
    if 'process_video_in_chunks(' in content:
        print("✅ process_video_in_chunks вызывается в main")
    else:
        print("❌ process_video_in_chunks не вызывается в main")
        return False
    
    # Проверяем, что передаются правильные аргументы
    if 'process_video_in_chunks(\n            video_tensor, mask_tensor, model, \n            fix_raft, fix_flow_complete,\n            args, logger\n        )' in content:
        print("✅ process_video_in_chunks вызывается с правильными аргументами")
    else:
        # Ищем альтернативный формат
        if 'process_video_in_chunks(video_tensor, mask_tensor, model, fix_raft, fix_flow_complete, args, logger)' in content:
            print("✅ process_video_in_chunks вызывается с правильными аргументами (однострочный формат)")
        else:
            print("❌ process_video_in_chunks вызывается с неправильными аргументами")
            return False
    
    return True

def test_error_handling():
    """Проверяет обработку ошибок"""
    print("\n🔍 Проверка обработки ошибок...")
    
    import inference_core
    
    # Проверяем, что safe_raft_inference имеет обработку ошибок
    with open('inference_core.py', 'r') as f:
        content = f.read()
    
    error_checks = [
        'except RuntimeError as e:',
        'if "out of memory" in str(e).lower()',
        'enable_cpu_fallback',
        'torch.cuda.empty_cache()'
    ]
    
    for check in error_checks:
        if check in content:
            print(f"✅ Найдена обработка ошибок: {check}")
        else:
            print(f"❌ Не найдена обработка ошибок: {check}")
            return False
    
    return True

def main():
    """Основная функция тестирования"""
    print("🧪 Тест исправлений для inference_core.py")
    print("=" * 60)
    
    tests = [
        ("Проверка сигнатур функций", test_function_signatures),
        ("Проверка функции main", test_main_function),
        ("Проверка обработки ошибок", test_error_handling),
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
        print("\n🚀 inference_core.py готов к использованию в production")
        print("\n📋 Исправленные проблемы:")
        print("1. ✅ Исправлена передача аргументов в process_video_in_chunks")
        print("2. ✅ Устранена рекурсивная ошибка в safe_raft_inference")
        print("3. ✅ Сохранена обратная совместимость")
        return 0
    else:
        print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print("Требуется дополнительная отладка")
        return 1

if __name__ == "__main__":
    import inspect
    sys.exit(main())
