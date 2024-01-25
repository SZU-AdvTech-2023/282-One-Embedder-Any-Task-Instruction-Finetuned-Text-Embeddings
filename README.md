运行test.py进行测试\
运行以下命令评估Billboard数据集
### Billboard
```python
cd evaluation/text_evaluation
python main.py --model_name hkunlp/instructor-large --task mscoco --add_prompt
```
您可以通过指定“--model_name”来评估经过训练的模型检查点，并通过更改“--task”来运行所有公告牌数据集。