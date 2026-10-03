"""
游戏内AI与机器人系统 - 核心模块（修复版）

五个子系统：
- 行为树 behavior_tree：巡逻/搜索/攻击/掩护/撤退等多种行为，带条件判断与状态记忆
- 感知系统 perception_system：视觉 + 听觉 + 位置记忆 + 可疑度
- 决策系统 decision_making：多种决策 + 优先级 + 考虑队友状态
- bot通信 bot_communication：信息共享 + 消息优先级 + 传输延迟
- 难度系统 difficulty_system：真实影响属性 / 反应时间 / 瞄准精度 / 行为复杂度
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random

Position = Tuple[float, float]


def _distance(a: Position, b: Position) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# 难度真实影响：属性、反应时间、瞄准精度、可用战术（行为复杂度）
DIFFICULTY_PROFILES: Dict[str, Dict] = {
    "easy": {
        "health": 60.0,
        "damage": 5.0,
        "speed": 2.0,
        "reaction_time": 0.9,
        "aim_accuracy": 0.30,
        "tactics": {"patrol", "attack", "search"},
    },
    "normal": {
        "health": 100.0,
        "damage": 10.0,
        "speed": 3.0,
        "reaction_time": 0.5,
        "aim_accuracy": 0.60,
        "tactics": {"patrol", "attack", "search", "take_cover"},
    },
    "hard": {
        "health": 150.0,
        "damage": 15.0,
        "speed": 4.0,
        "reaction_time": 0.2,
        "aim_accuracy": 0.90,
        "tactics": {"patrol", "attack", "search", "take_cover", "retreat", "flank"},
    },
}

# 通信消息优先级：重要信息优先投递
MESSAGE_PRIORITIES = {
    "enemy_spotted": 3,
    "request_backup": 2,
    "status": 1,
}

# 决策优先级（越靠前优先级越高）
DECISION_PRIORITIES = [
    "retreat",
    "reload",
    "take_cover",
    "flank",
    "shoot",
    "support",
    "search",
    "patrol",
]


@dataclass
class Bot:
    bot_id: str
    position: Position = (0.0, 0.0)
    facing: float = 0.0
    difficulty: str = "normal"
    health: float = 100.0
    max_health: float = 100.0
    ammo: int = 30
    behavior: str = "patrol"
    suspicion: float = 0.0
    blackboard: Dict = field(default_factory=dict)
    inbox: List[Dict] = field(default_factory=list)


@dataclass
class Player:
    player_id: str
    position: Position = (0.0, 0.0)
    alive: bool = True


class AIBotSystem:
    VIEW_DISTANCE = 30.0
    FIELD_OF_VIEW = math.radians(120.0)
    HEARING_RADIUS = 25.0
    MEMORY_DURATION = 5.0
    ALERT_THRESHOLD = 0.8
    RETREAT_HEALTH_RATIO = 0.3
    UNDER_FIRE_DURATION = 2.0
    VISION_SUSPICION_RATE = 2.0
    HEARING_SUSPICION_RATE = 1.0
    SUSPICION_DECAY_RATE = 0.1
    MAX_POSITION_ERROR = 5.0
    DEFAULT_COMMUNICATION_DELAY = 0.3

    BEHAVIORS = ("patrol", "search", "attack", "take_cover", "retreat", "alert")

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.difficulty = self.config.get("difficulty", "normal")
        if self.difficulty not in DIFFICULTY_PROFILES:
            raise ValueError(f"未知难度: {self.difficulty}")
        self.time = 0.0
        self.bots: Dict[str, Bot] = {}
        self.players: Dict[str, Player] = {}
        self._message_queue: List[Dict] = []
        self._sound_events: List[Dict] = []
        self._rng = random.Random(self.config.get("random_seed"))
        self._state = {}
        self._history = []

    # ---------- 实体管理 ----------

    def add_bot(self, bot_id: str, position: Position = (0.0, 0.0),
                facing: float = 0.0, difficulty: Optional[str] = None) -> Bot:
        diff = difficulty or self.difficulty
        profile = self.difficulty_system(diff)
        bot = Bot(bot_id=bot_id, position=tuple(position), facing=facing, difficulty=diff)
        bot.health = bot.max_health = profile["health"]
        bot.blackboard.update({
            "last_known_position": None,
            "last_known_time": -float("inf"),
            "spotted_time": None,
            "under_fire_time": -float("inf"),
            "engaging": False,
            "behavior_history": ["patrol"],
        })
        self.bots[bot_id] = bot
        return bot

    def add_player(self, player_id: str, position: Position = (0.0, 0.0)) -> Player:
        player = Player(player_id=player_id, position=tuple(position))
        self.players[player_id] = player
        return player

    def apply_damage(self, bot_id: str, amount: float) -> float:
        bot = self.bots[bot_id]
        bot.health = max(0.0, bot.health - amount)
        bot.blackboard["under_fire_time"] = self.time
        return bot.health

    def reset(self):
        self.time = 0.0
        self.bots = {}
        self.players = {}
        self._message_queue = []
        self._sound_events = []
        self._state = {}
        self._history = []

    def update(self, dt: float):
        self.time += dt
        for bot_id in list(self.bots):
            self.perception_system(bot_id, dt)
        for bot_id in list(self.bots):
            self.behavior_tree(bot_id)
        self._deliver_messages()
        self._sound_events.clear()

    # ---------- 难度系统 ----------

    def difficulty_system(self, level: Optional[str] = None) -> Dict:
        level = level or self.difficulty
        if level not in DIFFICULTY_PROFILES:
            raise ValueError(f"未知难度: {level}")
        profile = dict(DIFFICULTY_PROFILES[level])
        profile["tactics"] = set(profile["tactics"])
        return profile

    def hit_probability(self, bot_id: str, distance: Optional[float] = None) -> float:
        bot = self.bots[bot_id]
        accuracy = self.difficulty_system(bot.difficulty)["aim_accuracy"]
        if distance is not None:
            accuracy *= max(0.2, 1.0 - distance / (self.VIEW_DISTANCE * 2.0))
        return max(0.0, min(1.0, accuracy))

    def shoot(self, bot_id: str, target_id: str) -> bool:
        bot = self.bots[bot_id]
        target = self.players[target_id]
        if bot.ammo <= 0:
            return False
        bot.ammo -= 1
        dist = _distance(bot.position, target.position)
        return self._rng.random() < self.hit_probability(bot_id, dist)

    # ---------- 感知系统 ----------

    def emit_sound(self, position: Position, loudness: float = 1.0,
                   source: Optional[str] = None) -> Dict:
        event = {
            "position": tuple(position),
            "loudness": loudness,
            "source": source,
            "time": self.time,
        }
        self._sound_events.append(event)
        return event

    def can_see(self, bot: Bot, player: Player) -> bool:
        if not player.alive:
            return False
        if _distance(bot.position, player.position) > self.VIEW_DISTANCE:
            return False
        angle_to = math.atan2(
            player.position[1] - bot.position[1],
            player.position[0] - bot.position[0],
        )
        return abs(_normalize_angle(angle_to - bot.facing)) <= self.FIELD_OF_VIEW / 2.0

    def _visible_players(self, bot: Bot) -> List[Player]:
        return [p for p in self.players.values() if self.can_see(bot, p)]

    def _audible_events(self, bot: Bot) -> List[Dict]:
        return [
            e for e in self._sound_events
            if e["source"] != bot.bot_id
            and _distance(bot.position, e["position"]) <= self.HEARING_RADIUS * e["loudness"]
        ]

    def _memory_fresh(self, bot: Bot) -> bool:
        bb = bot.blackboard
        return (bb["last_known_position"] is not None
                and self.time - bb["last_known_time"] <= self.MEMORY_DURATION)

    def perception_system(self, bot_id: str, dt: float = 0.0) -> Dict:
        bot = self.bots[bot_id]
        bb = bot.blackboard
        visible = self._visible_players(bot)
        heard = self._audible_events(bot)

        if visible:
            bot.suspicion = min(1.0, bot.suspicion + self.VISION_SUSPICION_RATE * dt)
            if bb["spotted_time"] is None:
                bb["spotted_time"] = self.time - dt
            nearest = min(visible, key=lambda p: _distance(bot.position, p.position))
            bb["last_known_position"] = nearest.position
            bb["last_known_time"] = self.time
        elif heard:
            # 听觉：看不到也能察觉（如背后的枪声）
            bot.suspicion = min(1.0, bot.suspicion + self.HEARING_SUSPICION_RATE * dt)
            nearest = min(heard, key=lambda e: _distance(bot.position, e["position"]))
            bb["last_known_position"] = nearest["position"]
            bb["last_known_time"] = self.time
        else:
            bot.suspicion = max(0.0, bot.suspicion - self.SUSPICION_DECAY_RATE * dt)

        # 记忆：玩家离开视线后保留一段时间，超时遗忘
        if not self._memory_fresh(bot):
            bb["last_known_position"] = None
            bb["spotted_time"] = None

        return {
            "visible_players": [p.player_id for p in visible],
            "heard_events": heard,
            "suspicion": bot.suspicion,
            "alerted": bot.suspicion >= self.ALERT_THRESHOLD,
            "last_known_position": bb["last_known_position"],
            "memory_valid": self._memory_fresh(bot),
        }

    def estimated_player_position(self, bot_id: str) -> Optional[Position]:
        """bot 认为的玩家位置：完全警觉且可见时精确，否则带误差（可疑度越低误差越大）"""
        bot = self.bots[bot_id]
        bb = bot.blackboard
        visible = self._visible_players(bot)
        if visible and bot.suspicion >= self.ALERT_THRESHOLD:
            nearest = min(visible, key=lambda p: _distance(bot.position, p.position))
            return nearest.position
        if bb["last_known_position"] is None:
            return None
        error = (1.0 - bot.suspicion) * self.MAX_POSITION_ERROR
        lx, ly = bb["last_known_position"]
        return (lx + error, ly + error)

    def _has_reacted(self, bot: Bot) -> bool:
        spotted = bot.blackboard["spotted_time"]
        if spotted is None:
            return False
        reaction_time = self.difficulty_system(bot.difficulty)["reaction_time"]
        return self.time - spotted >= reaction_time

    # ---------- 行为树 ----------

    def behavior_tree(self, bot_id: str) -> str:
        """Selector 行为树：条件节点按优先级求值，blackboard 提供状态记忆"""
        bot = self.bots[bot_id]
        bb = bot.blackboard
        tactics = self.difficulty_system(bot.difficulty)["tactics"]
        visible = self._visible_players(bot)
        under_fire = self.time - bb["under_fire_time"] <= self.UNDER_FIRE_DURATION
        threat_known = bool(visible) or self._memory_fresh(bot)

        if (bot.health <= bot.max_health * self.RETREAT_HEALTH_RATIO
                and threat_known and "retreat" in tactics):
            behavior = "retreat"
        elif visible and not self._has_reacted(bot):
            behavior = "alert"  # 已察觉但反应时间未到，还不能行动
        elif visible and under_fire and "take_cover" in tactics:
            behavior = "take_cover"
        elif visible:
            behavior = "attack"
        elif self._memory_fresh(bot) and "search" in tactics:
            behavior = "search"
        else:
            behavior = "patrol"

        # 状态记忆：记录行为历史
        if bb["behavior_history"][-1] != behavior:
            bb["behavior_history"].append(behavior)
        bot.behavior = behavior
        bb["engaging"] = behavior == "attack"
        return behavior

    # ---------- 决策系统 ----------

    def _teammate_engaging(self, bot: Bot) -> bool:
        return any(
            other.blackboard.get("engaging")
            for other in self.bots.values()
            if other.bot_id != bot.bot_id
        )

    def decision_making(self, bot_id: str) -> str:
        """多种决策 + 优先级仲裁 + 考虑队友状态"""
        bot = self.bots[bot_id]
        bb = bot.blackboard
        tactics = self.difficulty_system(bot.difficulty)["tactics"]
        visible = self._visible_players(bot)
        under_fire = self.time - bb["under_fire_time"] <= self.UNDER_FIRE_DURATION
        threat_known = bool(visible) or self._memory_fresh(bot)
        teammate_engaging = self._teammate_engaging(bot)

        candidates = {"patrol"}
        if (bot.health <= bot.max_health * self.RETREAT_HEALTH_RATIO
                and threat_known and "retreat" in tactics):
            candidates.add("retreat")
        if visible and bot.ammo <= 0:
            candidates.add("reload")
        if visible and under_fire and "take_cover" in tactics:
            candidates.add("take_cover")
        if visible and bot.ammo > 0:
            # 队友已在交火时选择绕后，而不是各打各的
            if teammate_engaging and "flank" in tactics:
                candidates.add("flank")
            else:
                candidates.add("shoot")
        if not visible and teammate_engaging:
            candidates.add("support")
        if not visible and self._memory_fresh(bot):
            candidates.add("search")

        for decision in DECISION_PRIORITIES:
            if decision in candidates:
                return decision
        return "patrol"

    # ---------- bot通信 ----------

    def bot_communication(self, sender_id: str, content, msg_type: str = "status",
                          delay: Optional[float] = None) -> Dict:
        if msg_type not in MESSAGE_PRIORITIES:
            raise ValueError(f"未知消息类型: {msg_type}")
        if delay is None:
            delay = self.DEFAULT_COMMUNICATION_DELAY
        message = {
            "sender": sender_id,
            "type": msg_type,
            "content": content,
            "priority": MESSAGE_PRIORITIES[msg_type],
            "sent_at": self.time,
            "deliver_at": self.time + delay,
        }
        self._message_queue.append(message)
        return message

    def _deliver_messages(self) -> List[Dict]:
        due = [m for m in self._message_queue if m["deliver_at"] <= self.time]
        # 高优先级消息先投递
        due.sort(key=lambda m: (-m["priority"], m["sent_at"]))
        for message in due:
            self._message_queue.remove(message)
            for bot_id, bot in self.bots.items():
                if bot_id == message["sender"]:
                    continue
                bot.inbox.append(message)
                self._apply_message(bot, message)
        return due

    def _apply_message(self, bot: Bot, message: Dict):
        # 信息共享：队友看到玩家，本bot也获得位置
        if message["type"] == "enemy_spotted" and isinstance(message["content"], dict):
            position = message["content"].get("position")
            if position is not None:
                bot.blackboard["last_known_position"] = tuple(position)
                bot.blackboard["last_known_time"] = self.time
