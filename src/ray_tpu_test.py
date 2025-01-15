"""TODO: ricliu - DO NOT SUBMIT without either providing a detailed docstring or
removing it altogether.
"""


from unittest import mock
import pytest
import ray_tpu


def test_give_me_a_name():
   ray_tpu.init()

   tpu_resources = ray_tpu.available_resources()


#if __name__ == "__main__":
#  googletest.main()
