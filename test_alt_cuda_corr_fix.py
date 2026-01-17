#!/usr/bin/env python3
"""
Тест исправления alt_cuda_corr для стабильности на CUDA 12+.
Проверяет, что alt_cuda_corr отключен и корреляция работает с FP32.
"""

import sys
import os
import torch

# Добавляем текущую директорию в путь
sys.path.insert(0, os.getcwd())

def test_alt_cuda_corr_disabled():
    """Проверяет, что alt_cuda_corr отключен."""
    print("🔍 Проверка отключения alt_cuda_corr...")
    
    # Импортируем модуль corr
    try:
        from RAFT import corr
        print("✅ Модуль RAFT.corr успешно импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта RAFT.corr: {e}")
        return False
    
    # Проверяем, что alt_cuda_corr установлен в None
    if hasattr(corr, 'alt_cuda_corr'):
        if corr.alt_cuda_corr is None:
            print("✅ alt_cuda_corr установлен в None (отключен)")
        else:
            print(f"❌ alt_cuda_corr не None: {corr.alt_cuda_corr}")
            return False
    else:
        print("❌ alt_cuda_corr не определен в модуле")
        return False
    
    # Проверяем, что импорт закомментирован
    with open('RAFT/corr.py', 'r') as f:
        content = f.read()
    
    if '# FORCE DISABLE alt_cuda_corr for stability on CUDA 12+' in content:
        print("✅ Комментарий об отключении присутствует")
    else:
        print("❌ Комментарий об отключении отсутствует")
        return False
    
    if 'alt_cuda_corr = None' in content:
        print("✅ alt_cuda_corr явно установлен в None")
    else:
        print("❌ alt_cuda_corr не установлен в None")
        return False
    
    return True

def test_corr_method_fp32():
    """Проверяет, что метод CorrBlock.corr использует FP32 и contiguous memory."""
    print("\n🔍 Проверка метода CorrBlock.corr...")
    
    from RAFT.corr import CorrBlock
    
    # Создаем dummy тензоры
    batch, dim, ht, wd = 2, 64, 32, 32
    fmap1 = torch.randn(batch, dim, ht, wd)
    fmap2 = torch.randn(batch, dim, ht, wd)
    
    print(f"  Созданы тензоры: fmap1 shape {fmap1.shape}, dtype {fmap1.dtype}")
    print(f"                   fmap2 shape {fmap2.shape}, dtype {fmap2.dtype}")
    
    # Вызываем статический метод corr
    try:
        corr_result = CorrBlock.corr(fmap1, fmap2)
        print(f"✅ Метод CorrBlock.corr выполнен успешно")
        print(f"  Результат: shape {corr_result.shape}, dtype {corr_result.dtype}")
    except Exception as e:
        print(f"❌ Ошибка в CorrBlock.corr: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Проверяем, что результат в FP32
    if corr_result.dtype == torch.float32:
        print("✅ Результат в FP32 (как ожидалось)")
    else:
        print(f"❌ Результат не в FP32: {corr_result.dtype}")
        return False
    
    # Проверяем код метода на наличие патча
    with open('RAFT/corr.py', 'r') as f:
        content = f.read()
    
    if '# Force FP32 and Contiguous memory layout' in content:
        print("✅ Комментарий о FP32 и contiguous memory присутствует")
    else:
        print("❌ Комментарий о FP32 и contiguous memory отсутствует")
        return False
    
    if 'f1 = fmap1.float().transpose(1,2).contiguous()' in content:
        print("✅ Используется .float() и .contiguous() для f1")
    else:
        print("❌ Не используется .float() и .contiguous() для f1")
        return False
    
    if 'f2 = fmap2.float().contiguous()' in content:
        print("✅ Используется .float() и .contiguous() для f2")
    else:
        print("❌ Не используется .float() и .contiguous() для f2")
        return False
    
    return True

def test_alternate_corr_block():
    """Проверяет, что AlternateCorrBlock выдает понятную ошибку."""
    print("\n🔍 Проверка AlternateCorrBlock...")
    
    from RAFT.corr import AlternateCorrBlock
    
    # Создаем dummy тензоры
    batch, dim, ht, wd = 1, 32, 16, 16
    fmap1 = torch.randn(batch, dim, ht, wd)
    fmap2 = torch.randn(batch, dim, ht, wd)
    
    print(f"  Созданы тензоры для AlternateCorrBlock")
    
    # Создаем экземпляр AlternateCorrBlock
    try:
        alt_block = AlternateCorrBlock(fmap1, fmap2, num_levels=2, radius=2)
        print("✅ AlternateCorrBlock инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации AlternateCorrBlock: {e}")
        return False
    
    # Пытаемся вызвать __call__ - должен вызвать RuntimeError
    coords = torch.randn(batch, 2, ht, wd)
    try:
        result = alt_block(coords)
        print(f"❌ AlternateCorrBlock не вызвал ошибку (возможно, alt_cuda_corr доступен)")
        print(f"  Результат: shape {result.shape}")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        print(f"✅ AlternateCorrBlock вызвал RuntimeError (как ожидалось)")
        print(f"  Сообщение: {error_msg}")
        if 'disabled for stability' in error_msg or 'alt_cuda_corr' in error_msg:
            print("✅ Сообщение об ошибке указывает на отключение alt_cuda_corr")
        else:
            print("⚠️  Сообщение об ошибке не указывает на отключение alt_cuda_corr")
    except Exception as e:
        print(f"❌ AlternateCorrBlock вызвал неожиданную ошибку: {e}")
        return False
    
    return True

def test_corr_block_integration():
    """Проверяет интеграцию CorrBlock (создание экземпляра и вызов)."""
    print("\n🔍 Проверка интеграции CorrBlock...")
    
    from RAFT.corr import CorrBlock
    
    batch, dim, ht, wd = 1, 64, 32, 32
    fmap1 = torch.randn(batch, dim, ht, wd)
    fmap2 = torch.randn(batch, dim, ht, wd)
    
    print(f"  Созданы тензоры для CorrBlock")
    
    try:
        corr_block = CorrBlock(fmap1, fmap2, num_levels=2, radius=2)
        print("✅ CorrBlock инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации CorrBlock: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Проверяем вызов __call__
    coords = torch.randn(batch, 2, ht, wd)
    try:
        result = corr_block(coords)
        print(f"✅ CorrBlock.__call__ выполнен успешно")
        print(f"  Результат: shape {result.shape}, dtype {result.dtype}")
    except Exception as e:
        print(f"❌ Ошибка в CorrBlock.__call__: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Основная функция тестирования."""
    print("🧪 Тест исправления alt_cuda_corr для стабильности на CUDA 12+")
    print("=" * 60)
    
    tests = [
        ("Отключение alt_cuda_corr", test_alt_cuda_corr_disabled),
        ("Проверка метода CorrBlock.corr", test_corr_method_fp32),
        ("Проверка AlternateCorrBlock", test_alternate_corr_block),
        ("Интеграция CorrBlock", test_corr_block_integration),
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
        print("\n📋 Исправления успешно применены:")
        print("1. ✅ alt_cuda_corr принудительно отключен (установлен в None)")
        print("2. ✅ Метод CorrBlock.corr использует FP32 и contiguous memory")
        print("3. ✅ AlternateCorrBlock выдает понятную ошибку при использовании")
        print("4. ✅ CorrBlock работает корректно с исправлениями")
        print("\n🚀 Исправление готово к использованию в production")
        return 0
    else:
        print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print("Требуется дополнительная отладка")
        return 1

if __name__ == "__main__":
    sys.exit(main())
