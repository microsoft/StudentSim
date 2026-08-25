"""Per-domain pieces: prompts, profiles, guidance modes, and metrics.

There is no domain object. Training takes the same code path in every domain
and differs only in the values its configuration carries, and evaluation looks
up what it needs by domain name. Import the piece you want from
``studentsim.domains.<domain>``.
"""
