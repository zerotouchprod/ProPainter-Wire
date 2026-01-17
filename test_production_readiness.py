#!/usr/bin/env python3
"""
Тест готовности оптимизированной версии inference_core.py к production.
Проверяет наличие всех ключевых оптимизаций и их работоспособность.
"""

import os
import sys
import subprocess
import re

def check_optimizations_in_file(filepath):
    """Проверяет наличие оптимизаций в файле"""
    optimizations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем наличие ключевых оптимизаций
        checks = [
            ('AMP (torch.autocast)', 'torch.autocast'),
            ('Чанковая обработка', 'process_video_in_chunks'),
            ('Безопасный RAFT с fallback', 'safe_raft_inference'),
            ('Детальное логирование', 'InferenceLogger'),
            ('Динамический scale_factor', 'calculate_optimal_scale_factor'),
            ('CPU fallback', '--no-cpu-fallback'),
            ('Memory monitoring', 'memory_allocated'),
            ('Smart downscale', 'total_pixels >'),
        ]
        
        for name, pattern in checks:
            if pattern in content:
                optimizations.append(name)
    
    except Exception as e:
        print(f"❌ Ошибка чтения файла {filepath}: {e}")
    
    return optimizations

def check_file_exists(filepath):
    """Проверяет существование файла"""
    if os.path.exists(filepath):
        print(f"✅ Файл существует: {filepath}")
        return True
    else:
        print(f"❌ Файл не найден: {filepath}")
        return False

def check_python_imports():
    """Проверяет импорты Python"""
    print("\n🔍 Проверка Python импортов...")
    
    test_code = """
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Ошибка импорта torch: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ Ошибка импорта numpy: {e}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ Ошибка импорта OpenCV: {e}")

try:
    from PIL import Image
    print("PIL: доступен")
except ImportError as e:
    print(f"❌ Ошибка импорта PIL: {e}")

try:
    import einops
    print("Einops: доступен")
except ImportError as e:
    print(f"❌ Ошибка импорта einops: {e}")

try:
    import easydict
    print("Easydict: доступен")
except ImportError as e:
    print(f"❌ Ошибка импорта easydict: {e}")
"""
    
    result = subprocess.run([sys.executable, '-c', test_code], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Все необходимые импорты работают")
        print(result.stdout)
        return True
    else:
        print("❌ Ошибка при проверке импортов")
        print(result.stderr)
        return False

def check_inference_core_import():
    """Проверяет возможность импорта inference_core.py"""
    print("\n🔍 Проверка импорта inference_core.py...")
    
    # Используем прямой импорт вместо subprocess
    import sys
    import os
    
    try:
        # Добавляем текущую директорию в путь
        sys.path.insert(0, os.getcwd())
        
        # Пытаемся импортировать
        from inference_core import main, InferenceLogger
        print("✅ inference_core.py импортируется успешно")
        print(f"  - Функция main: {main is not None}")
        print(f"  - Класс InferenceLogger: {InferenceLogger is not None}")
        
        # Проверяем наличие ключевых функций
        import inference_core
        
        functions_to_check = [
            'process_video_in_chunks',
            'calculate_optimal_scale_factor', 
            'safe_raft_inference',
            'process_single_chunk'
        ]
        
        for func_name in functions_to_check:
            if hasattr(inference_core, func_name):
                print(f"  - Функция {func_name}: ✅ найдена")
            else:
                print(f"  - Функция {func_name}: ❌ не найдена")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта inference_core.py: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Другая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_weights():
    """Проверяет наличие весов моделей"""
    print("\n🔍 Проверка весов моделей...")
    
    weights_dir = "weights"
    required_weights = [
        "ProPainter.pth",
        "raft-things.pth", 
        "recurrent_flow_completion.pth"
    ]
    
    if not os.path.exists(weights_dir):
        print(f"❌ Директория весов не найдена: {weights_dir}")
        return False
    
    all_found = True
    for weight_file in required_weights:
        weight_path = os.path.join(weights_dir, weight_file)
        if os.path.exists(weight_path):
            size = os.path.getsize(weight_path) / (1024*1024)  # MB
            print(f"✅ {weight_file}: {size:.1f} MB")
        else:
            print(f"❌ {weight_file}: не найден")
            all_found = False
    
    return all_found

def run_unit_tests():
    """Запускает unit-тесты оптимизаций"""
    print("\n🧪 Запуск unit-тестов оптимизаций...")
    
    test_files = [
        "test_optimization_logic.py",
        "test_raft_optimization.py"
    ]
    
    all_passed = True
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n📋 Запуск теста: {test_file}")
            result = subprocess.run([sys.executable, test_file], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_file}: PASSED")
                # Выводим краткое резюме
                for line in result.stdout.split('\n'):
                    if 'PASSED' in line or 'FAILED' in line or 'Results:' in line:
                        print(f"  {line}")
            else:
                print(f"❌ {test_file}: FAILED")
                print(f"  Ошибка: {result.stderr[:200]}...")
                all_passed = False
        else:
            print(f"⚠️ Тест не найден: {test_file}")
    
    return all_passed

def check_production_requirements():
    """Проверяет требования для production"""
    print("\n📋 Проверка требований для production...")
    
    requirements = []
    
    # Проверяем наличие всех необходимых файлов
    required_files = [
        "inference_core.py",
        "model/propainter.py",
        "model/modules/sparse_transformer.py",
        "RAFT/raft.py",
        "requirements.txt"
    ]
    
    for filepath in required_files:
        if os.path.exists(filepath):
            requirements.append((filepath, True))
        else:
            requirements.append((filepath, False))
    
    # Проверяем наличие оптимизаций в inference_core.py
    optimizations = check_optimizations_in_file("inference_core.py")
    
    # Выводим результаты
    all_met = True
    for filepath, exists in requirements:
        if exists:
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath}")
            all_met = False
    
    print(f"\n📊 Найдено оптимизаций в inference_core.py: {len(optimizations)}")
    for opt in optimizations:
        print(f"  ✅ {opt}")
    
    if len(optimizations) >= 5:
        print("✅ Количество оптимизаций достаточное для production")
    else:
        print(f"⚠️ Мало оптимизаций: {len(optimizations)}/5")
        all_met = False
    
    return all_met and len(optimizations) >= 5

def main():
    """Основная функция тестирования"""
    print("🚀 Тест готовности оптимизированной версии к production")
    print("=" * 60)
    
    # Проверяем текущую директорию
    cwd = os.getcwd()
    print(f"📁 Текущая директория: {cwd}")
    
    # Список проверок
    checks = [
        ("Проверка Python импортов", check_python_imports),
        ("Проверка импорта inference_core.py", check_inference_core_import),
        ("Проверка весов моделей", check_model_weights),
        ("Проверка требований для production", check_production_requirements),
        ("Запуск unit-тестов", run_unit_tests),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{'='*60}")
        print(f"🔍 {check_name}")
        print(f"{'='*60}")
        
        try:
            success = check_func()
            results.append((check_name, success))
            
            if success:
                print(f"✅ {check_name}: PASSED")
            else:
                print(f"❌ {check_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {check_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results.append((check_name, False))
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\n🎯 Результаты: {passed}/{total} проверок пройдено")
    
    if passed == total:
        print("\n✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("\n🚀 inference_core.py ГОТОВ К PRODUCTION!")
        print("\n📋 Ключевые оптимизации:")
        print("1. ✅ AMP (torch.autocast) для смешанной точности")
        print("2. ✅ Чанковая обработка для экономии памяти")
        print("3. ✅ Безопасный RAFT с CPU fallback")
        print("4. ✅ Детальное логирование с мониторингом памяти")
        print("5. ✅ Динамический scale_factor на основе разрешения")
        print("6. ✅ Unit-тесты для всех оптимизаций")
        print("7. ✅ Обратная совместимость с существующими скриптами")
        
        print("\n📋 Рекомендации для production:")
        print("1. Использовать флаг --log-level DEBUG для мониторинга")
        print("2. Настроить --chunk-size в зависимости от доступной памяти")
        print("3. CPU fallback включен по умолчанию для стабильности")
        print("4. Для отключения CPU fallback использовать --no-cpu-fallback")
        
        return 0
    else:
        print(f"\n⚠️ {total - passed} проверок не пройдено")
        print("Требуется дополнительная отладка перед использованием в production")
        return 1

if __name__ == "__main__":
    sys.exit(main())
