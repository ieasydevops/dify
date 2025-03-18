import pytest
from unittest.mock import Mock, patch

from models.dataset import Dataset, ChildChunk
from services.vector_service import VectorService


@pytest.fixture
def mock_dataset():
    dataset = Mock(spec=Dataset)
    dataset.indexing_technique = "high_quality"
    return dataset


@pytest.fixture
def mock_child_chunk():
    chunk = Mock(spec=ChildChunk)
    chunk.index_node_id = "test_id"
    chunk.content = "test content"
    chunk.index_node_hash = "test_hash"
    chunk.document_id = "doc_id"
    chunk.dataset_id = "dataset_id"
    return chunk


def test_update_child_chunk_vector_high_quality(mock_dataset, mock_child_chunk):
    # 准备测试数据
    new_chunks = [mock_child_chunk]
    update_chunks = [mock_child_chunk]
    delete_chunks = [mock_child_chunk]

    # Mock Vector 类
    with patch('services.vector_service.Vector') as mock_vector:
        vector_instance = Mock()
        mock_vector.return_value = vector_instance

        # 调用测试方法
        VectorService.update_child_chunk_vector(
            new_child_chunks=new_chunks,
            update_child_chunks=update_chunks,
            delete_child_chunks=delete_chunks,
            dataset=mock_dataset
        )

        # 验证调用
        vector_instance.delete_by_ids.assert_called_once_with([mock_child_chunk.index_node_id])
        assert vector_instance.add_texts.call_count == 1
        documents = vector_instance.add_texts.call_args[0][0]
        assert len(documents) == 2  # 新增和更新的文档


def test_update_child_chunk_vector_non_high_quality(mock_dataset, mock_child_chunk):
    # 设置非高质量索引
    mock_dataset.indexing_technique = "low_quality"

    # 准备测试数据
    new_chunks = [mock_child_chunk]
    update_chunks = [mock_child_chunk]
    delete_chunks = [mock_child_chunk]

    # Mock Vector 类
    with patch('services.vector_service.Vector') as mock_vector:
        vector_instance = Mock()
        mock_vector.return_value = vector_instance

        # 调用测试方法
        VectorService.update_child_chunk_vector(
            new_child_chunks=new_chunks,
            update_child_chunks=update_chunks,
            delete_child_chunks=delete_chunks,
            dataset=mock_dataset
        )

        # 验证没有调用 Vector 方法
        vector_instance.delete_by_ids.assert_not_called()
        vector_instance.add_texts.assert_not_called()


def test_update_child_chunk_vector_empty_lists(mock_dataset):
    # 测试空列表的情况
    with patch('services.vector_service.Vector') as mock_vector:
        vector_instance = Mock()
        mock_vector.return_value = vector_instance

        VectorService.update_child_chunk_vector(
            new_child_chunks=[],
            update_child_chunks=[],
            delete_child_chunks=[],
            dataset=mock_dataset
        )

        # 验证没有调用 Vector 方法
        vector_instance.delete_by_ids.assert_not_called()
        vector_instance.add_texts.assert_not_called() 