import json

from typing import Any, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryConfig:
    """
    Data class to represent the different aspects of a synthetic dataset category.

    label is the title of the category
    starting points are the Wikipedia (list) articles that are used as the starting point
    for sampling articles for the category.
    category_pattern, explicit_category and title_prefix are helper variables for matching
    articles to a category during the scraping process.
    """
    label: str
    starting_points: list[str]
    category_pattern: Optional[str] = None
    explicit_category: Optional[str] = None
    title_prefix: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'CategoryConfig':
        return cls(**config)

    def __hash__(self):
        return hash(self.label)


@dataclass(frozen=True)
class Data:
    """
    Data class to represent the different aspects of a Wikipedia article.

    title represents the title of the article.
    text contains the summary, the first section, of an article.
    label states the category to which the article belongs to.
    """
    id: int
    question: str
    reference: str
    dense_ctxs: list[str]
    reranked_dense_ctxs: list[str]
    wrong_answer: str
    ori_fake: list[str]
    ori_fake_truthful_scores: list[str]
    reranked_dense_ctxs_truthful_scores: list[str]

    @classmethod
    def from_dict(cls, article: dict[str, str]) -> 'Data':
        article['wrong_answer'] = article.pop('wrong answer')
        return cls(**article)

    def __repr__(self):
        return f'<Article {self.title}>'


def load_data(file_path: str) -> list[Data]:
    """
    Load data from input JSON file.

    :param file_path: Path to the input JSON file.
    :type file_path: str
    :return: List of data.
    :rtype: list[Data]
    """
    with open(file_path, 'r') as file:
        json_data: list[dict[str, str]] = json.load(file)

    return list(map(Data.from_dict, json_data))
