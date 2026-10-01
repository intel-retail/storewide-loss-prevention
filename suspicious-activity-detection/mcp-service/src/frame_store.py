"""Read frame-object references copied into SeaweedFS for an alert."""

from __future__ import annotations

from typing import Any


class SeaweedFrameStore:
    def __init__(self, endpoint: str, bucket: str = "alerts", client: Any = None) -> None:
        self.endpoint = endpoint
        self.bucket = bucket
        if client is None:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                config=Config(signature_version=UNSIGNED),
            )
        self._client = client

    def find_alert_frames(self, object_id: str, alert_id: str) -> list[str]:
        if not object_id or not alert_id:
            return []
        prefix = f"{object_id}/{alert_id}/frames/"
        paginator = self._client.get_paginator("list_objects_v2")
        keys = [
            item["Key"]
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            for item in page.get("Contents", [])
            if item.get("Key", "").lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        return [f"s3://{self.bucket}/{key}" for key in sorted(keys)]