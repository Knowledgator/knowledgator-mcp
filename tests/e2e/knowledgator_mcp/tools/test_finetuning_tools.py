import asyncio
import pytest
from knowledgator import exceptions
from knowledgator_mcp.tools import dataset_tools, finetuning_tools


@pytest.mark.asyncio
async def test_list_trainings():
    """Verify list_trainings MCP tool against live API."""
    result = await finetuning_tools.list_trainings(limit=5)
    assert isinstance(result, (dict, list))


@pytest.mark.asyncio
async def test_get_training_status_existing():
    """Verify get_training_status with an existing training job ID if present."""
    result = await finetuning_tools.list_trainings(limit=5)
    trainings = []
    if isinstance(result, dict):
        trainings = result.get("trainings") or result.get("results") or result.get("data") or []
    elif isinstance(result, list):
        trainings = result

    if trainings and isinstance(trainings[0], dict):
        first = trainings[0]
        training_id = first.get("id") or first.get("training_id")
        if training_id:
            status_res = await finetuning_tools.get_training_status(training_id)
            assert isinstance(status_res, (dict, list))


@pytest.mark.asyncio
async def test_start_adapter_training_job():
    """Verify start_training_job for adapter training using working JSON schema."""
    # TODO add helper to create testing dataset if not exists
    placeholder_dataset_id = "6babd48a-2256-4466-b16b-9fd45ba59688"
    try:
        start_res = await finetuning_tools.start_training_job(
            model_id="comprehend-it",
            dataset_id=placeholder_dataset_id,
            type="adapter",
            hyperparameters={
                "seed": 0,
                "batch": None,
                "learning_rate_mult": None,
                "num_of_epochs": None,
            },
            validation_dataset_id=placeholder_dataset_id,
        )
        print(f"[DEBUG] start_adapter_res: {start_res}")
        assert isinstance(start_res, dict)
        training_id = start_res.get("id") or start_res.get("training_id")
        print(f"[DEBUG] adapter_training_id: {training_id}")

        if training_id:
            await asyncio.sleep(2)
            stop_res = await finetuning_tools.stop_training_job(training_id)
            print(f"[DEBUG] stop_adapter_res: {stop_res}")
            assert stop_res is None or isinstance(stop_res, (dict, list))

            status_res = await finetuning_tools.get_training_status(training_id)
            print(f"[DEBUG] status_adapter_res: {status_res}")
            assert isinstance(status_res, (dict, list))
    except exceptions.ClientError as exc:
        print(f"[DEBUG] test_start_adapter_training_job caught ClientError: {exc} (payload={getattr(exc, 'payload', None)})")
        if exc.status_code in (400, 403, 404):
            pytest.skip(f"Start adapter training job returned {exc.status_code} - skipping live test ({exc})")
        raise


@pytest.mark.asyncio
async def test_start_full_model_training_job():
    """Verify start_training_job for full-model training using working JSON schema."""
    # TODO add helper to create testing dataset if not exists
    placeholder_dataset_id = "6babd48a-2256-4466-b16b-9fd45ba59688"
    try:
        start_res = await finetuning_tools.start_training_job(
            model_id="gliclass-x-base",
            dataset_id=placeholder_dataset_id,
            type="model",
            hyperparameters={
                "seed": 0,
                "batch": None,
                "learning_rate_mult": None,
                "num_of_epochs": None,
            },
            validation_dataset_id=placeholder_dataset_id,
        )
        print(f"[DEBUG] start_full_model_res: {start_res}")
        assert isinstance(start_res, dict)
        training_id = start_res.get("id") or start_res.get("training_id")
        print(f"[DEBUG] full_model_training_id: {training_id}")

        if training_id:
            await asyncio.sleep(2)
            stop_res = await finetuning_tools.stop_training_job(training_id)
            print(f"[DEBUG] stop_full_model_res: {stop_res}")
            assert stop_res is None or isinstance(stop_res, (dict, list))

            status_res = await finetuning_tools.get_training_status(training_id)
            print(f"[DEBUG] status_full_model_res: {status_res}")
            assert isinstance(status_res, (dict, list))
    except exceptions.ClientError as exc:
        print(f"[DEBUG] test_start_full_model_training_job caught ClientError: {exc} (payload={getattr(exc, 'payload', None)})")
        if exc.status_code in (400, 403, 404):
            pytest.skip(f"Start full-model training job returned {exc.status_code} - skipping live test ({exc})")
        raise


