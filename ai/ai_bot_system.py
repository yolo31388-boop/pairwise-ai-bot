"""
游戏内AI与机器人系统 - 核心模块（修复版）

子系统：
- BehaviorTree      行为树：多种行为节点 + 条件判断 + 状态记忆（黑板）
- PerceptionSystem  感知系统：视线 + 听觉 + 记忆 + 可疑度
- DecisionMaking    决策系统：多种决策 + 优先级 + 队友状态感知
- BotCommunication  bot通信：信息共享 + 优先级 + 传递延迟
- DifficultySystem  难度系统：真实影响属性 / 反应时间 / 瞄准精度 / 行为复杂度
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import math
import time


@dataclass
class Config:
    pass


Position = Tuple[float, float]


def _distance(a: Position, b: Position) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_to(a: Position, b: Position) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


# ---------------------------------------------------------------- 实体

@dataclass
class Bot:
    bot_id: str
    position: Position = (0.0, 0.0)
    difficulty: str = "normal"
    team: str = "bots"
    facing: float = 0.0                 # 朝向（弧度，0 = +x 方向）
    health: float = 100.0
    max_health: float = 100.0
    speed: float = 4.0
    ammo: int = 30
    max_ammo: int = 30
    reaction_time: float = 0.5          # 由难度系统写入
    aim_accuracy: float = 0.55          # 由难度系统写入
    tactics: List[str] = field(default_factory=list)  # 由难度系统写入
    state: str = "idle"
    suspicion: float = 0.0              # 可疑度 0~threshold
    detected_at: Optional[float] = None # 首次确认敌人的时刻（反应时间起点）
    last_known_player_pos: Optional[Position] = None  # 感知记忆
    last_seen_time: Optional[float] = None
    blackboard: Dict = field(default_factory=dict)    # 行为树状态记忆


# ---------------------------------------------------------------- 难度系统

@dataclass
class DifficultyPreset:
    name: str
    health: float
    speed: float
    reaction_time: float   # 秒，越小反应越快
    aim_accuracy: float    # 0~1，基础命中率
    tactics: List[str]     # 行为复杂度：该难度可用的战术


class DifficultySystem:
    """难度真实影响 bot 的属性、反应时间、瞄准精度与行为复杂度。"""

    PRESETS = {
        "easy": DifficultyPreset(
            "easy", health=80.0, speed=3.0, reaction_time=0.8, aim_accuracy=0.30,
            tactics=["patrol", "engage", "reload"],
        ),
        "normal": DifficultyPreset(
            "normal", health=100.0, speed=4.0, reaction_time=0.5, aim_accuracy=0.55,
            tactics=["patrol", "engage", "reload", "take_cover", "search"],
        ),
        "hard": DifficultyPreset(
            "hard", health=120.0, speed=5.0, reaction_time=0.2, aim_accuracy=0.85,
            tactics=["patrol", "engage", "reload", "take_cover", "search", "retreat", "flank"],
        ),
    }

    def get_preset(self, difficulty: str) -> DifficultyPreset:
        if difficulty not in self.PRESETS:
            raise ValueError(f"未知难度: {difficulty}")
        return self.PRESETS[difficulty]

    def apply(self, bot: Bot) -> None:
        preset = self.get_preset(bot.difficulty)
        bot.max_health = preset.health
        bot.health = preset.health
        bot.speed = preset.speed
        bot.reaction_time = preset.reaction_time
        bot.aim_accuracy = preset.aim_accuracy
        bot.tactics = list(preset.tactics)

    def hit_probability(self, bot: Bot, distance: float, view_range: float = 30.0) -> float:
        """命中率 = 基础精度 × 距离衰减。困难 bot 显著更准。"""
        falloff = max(0.1, 1.0 - distance / max(view_range, 1e-6))
        return bot.aim_accuracy * falloff


# ---------------------------------------------------------------- 感知系统

class PerceptionSystem:
    """视线 + 听觉 + 记忆 + 可疑度。"""

    def __init__(self, view_range: float = 30.0, fov: float = math.radians(120.0),
                 hearing_range: float = 25.0, memory_duration: float = 5.0,
                 suspicion_threshold: float = 1.0):
        self.view_range = view_range
        self.fov = fov
        self.hearing_range = hearing_range
        self.memory_duration = memory_duration
        self.suspicion_threshold = suspicion_threshold
        self.vision_rate = 3.0    # 看到玩家时可疑度增速（/秒）
        self.hearing_rate = 1.5   # 听到声音时可疑度增速（/秒）
        self.decay_rate = 0.5     # 无刺激时可疑度衰减（/秒）

    def can_see(self, bot: Bot, target_pos: Position) -> bool:
        """视线检测：受距离与视野角限制。"""
        if _distance(bot.position, target_pos) > self.view_range:
            return False
        angle = _angle_to(bot.position, target_pos)
        diff = abs((angle - bot.facing + math.pi) % (2 * math.pi) - math.pi)
        return diff <= self.fov / 2

    def can_hear(self, bot: Bot, noise_pos: Position, noise_radius: Optional[float] = None) -> bool:
        """听觉检测：无方向性，背后的枪声也能听到。"""
        radius = noise_radius if noise_radius is not None else self.hearing_range
        return _distance(bot.position, noise_pos) <= radius

    def is_aware(self, bot: Bot) -> bool:
        """可疑度超过阈值才确认敌人准确位置。"""
        return bot.suspicion >= self.suspicion_threshold

    def has_reacted(self, bot: Bot, now: float) -> bool:
        """反应时间由难度决定：确认敌人后需等待 reaction_time 才能行动。"""
        if bot.detected_at is None:
            return False
        return (now - bot.detected_at) >= bot.reaction_time

    def recall(self, bot: Bot, now: float) -> Optional[Position]:
        """感知记忆：玩家离开视线后仍记得最后位置，超过 memory_duration 才遗忘。"""
        if bot.last_known_player_pos is None or bot.last_seen_time is None:
            return None
        if now - bot.last_seen_time > self.memory_duration:
            return None
        return bot.last_known_player_pos

    def update(self, bot: Bot, player_pos: Optional[Position], now: float, dt: float,
               noise_events: Optional[List[Position]] = None) -> None:
        seen = player_pos is not None and self.can_see(bot, player_pos)
        heard_pos = None
        for noise_pos in (noise_events or []):
            if self.can_hear(bot, noise_pos):
                heard_pos = noise_pos
                break

        if seen:
            bot.suspicion = min(self.suspicion_threshold,
                                bot.suspicion + self.vision_rate * dt)
        elif heard_pos is not None:
            bot.suspicion = min(self.suspicion_threshold,
                                bot.suspicion + self.hearing_rate * dt)
        else:
            bot.suspicion = max(0.0, bot.suspicion - self.decay_rate * dt)

        if self.is_aware(bot):
            if bot.detected_at is None:
                bot.detected_at = now
            if seen:
                bot.last_known_player_pos = player_pos
                bot.last_seen_time = now
            elif heard_pos is not None:
                # 听觉只能给出大致方位
                bot.last_known_player_pos = heard_pos
                bot.last_seen_time = now


# ---------------------------------------------------------------- 决策系统

class DecisionMaking:
    """多种决策 + 优先级（rules 按优先级从高到低排列）+ 考虑队友状态。"""

    def __init__(self):
        self.rules = [
            ("retreat",    lambda b, c: b.health <= 0.25 * b.max_health
                                        and "retreat" in b.tactics),
            ("reload",     lambda b, c: b.ammo <= 0),
            ("take_cover", lambda b, c: c.get("under_fire", False)
                                        and b.health <= 0.6 * b.max_health
                                        and "take_cover" in b.tactics),
            ("flank",      lambda b, c: c.get("player_visible", False)
                                        and c.get("flank_opportunity", False)
                                        and "flank" in b.tactics),
            ("engage",     lambda b, c: c.get("player_visible", False)),
            ("support",    lambda b, c: c.get("teammate_engaged", False)),
            ("search",     lambda b, c: c.get("has_memory", False)),
            ("patrol",     lambda b, c: True),
        ]
        self.PRIORITIES = [name for name, _ in self.rules]

    def decide(self, bot: Bot, context: Dict) -> str:
        for name, condition in self.rules:
            if condition(bot, context):
                return name
        return "patrol"


# ---------------------------------------------------------------- 行为树

@dataclass
class BehaviorNode:
    name: str
    condition: object  # Callable[[Bot, Dict], bool]


class BehaviorTree:
    """多行为节点的行为树：tick 时按优先级评估条件，黑板保存状态记忆。"""

    def __init__(self, decision_making: DecisionMaking):
        self.decision_making = decision_making
        self.nodes = [BehaviorNode(name, cond) for name, cond in decision_making.rules]

    def tick(self, bot: Bot, context: Dict) -> str:
        for node in self.nodes:
            if node.condition(bot, context):
                bot.state = node.name
                bot.blackboard["last_action"] = node.name
                bot.blackboard["last_tick_state"] = node.name
                if context.get("player_pos") is not None:
                    bot.blackboard["last_player_pos"] = context["player_pos"]
                return node.name
        return "patrol"


# ---------------------------------------------------------------- bot通信

@dataclass
class Message:
    sender_id: str
    msg_type: str
    payload: Dict
    priority: int            # 越大越重要
    created_at: float
    deliver_at: float        # 传递延迟：到该时刻才可被接收
    delivered_to: Set[str] = field(default_factory=set)


class BotCommunication:
    """bot 间共享信息：消息带优先级与传递延迟，非瞬时、非无差别。"""

    PRIORITY_LEVELS = {
        "enemy_sighting": 3,   # 发现敌人：最高优先级
        "request_backup": 2,   # 请求支援
        "status_update": 1,    # 常规状态：最低优先级
    }

    def __init__(self, base_delay: float = 0.5, message_ttl: float = 10.0):
        self.base_delay = base_delay
        self.message_ttl = message_ttl
        self._queue: List[Message] = []

    def send(self, sender_id: str, msg_type: str, payload: Dict, now: float,
             priority: Optional[int] = None, delay: Optional[float] = None) -> Message:
        if priority is None:
            priority = self.PRIORITY_LEVELS.get(msg_type, 1)
        if delay is None:
            delay = self.base_delay
        msg = Message(sender_id=sender_id, msg_type=msg_type, payload=dict(payload),
                      priority=priority, created_at=now, deliver_at=now + delay)
        self._queue.append(msg)
        return msg

    def receive(self, bot_id: str, now: float) -> List[Message]:
        """只投递已到时的消息，按优先级从高到低排序。"""
        delivered = []
        for msg in self._queue:
            if msg.deliver_at <= now and msg.sender_id != bot_id and bot_id not in msg.delivered_to:
                msg.delivered_to.add(bot_id)
                delivered.append(msg)
        delivered.sort(key=lambda m: -m.priority)
        return delivered

    def purge(self, now: float) -> None:
        self._queue = [m for m in self._queue if now - m.created_at <= self.message_ttl]


# ---------------------------------------------------------------- 主系统

class AIBotSystem:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self.difficulty_system = DifficultySystem()
        self.perception_system = PerceptionSystem()
        self.decision_making = DecisionMaking()
        self.bot_communication = BotCommunication()
        self.behavior_tree = BehaviorTree(self.decision_making)
        self.bots: Dict[str, Bot] = {}
        self.player_pos: Optional[Position] = None
        self.noise_events: List[Position] = []
        self.time = 0.0

    # -- bot 管理 ------------------------------------------------------

    def add_bot(self, bot_id: str, difficulty: str = "normal",
                position: Position = (0.0, 0.0)) -> Bot:
        bot = Bot(bot_id=bot_id, difficulty=difficulty, position=position)
        self.difficulty_system.apply(bot)
        self.bots[bot_id] = bot
        return bot

    # -- 主循环 --------------------------------------------------------

    def _build_context(self, bot: Bot) -> Dict:
        perception = self.perception_system
        player_visible = (
            self.player_pos is not None
            and perception.can_see(bot, self.player_pos)
            and perception.is_aware(bot)
            and perception.has_reacted(bot, self.time)
        )
        teammate_engaged = any(
            other.state in ("engage", "take_cover", "flank")
            for other in self.bots.values()
            if other.bot_id != bot.bot_id
        )
        return {
            "player_visible": player_visible,
            "player_pos": self.player_pos if player_visible else None,
            "under_fire": bot.blackboard.get("under_fire", False),
            "flank_opportunity": bot.blackboard.get("flank_opportunity", False),
            "teammate_engaged": teammate_engaged,
            "has_memory": perception.recall(bot, self.time) is not None,
        }

    def update(self, dt: float):
        self.time += dt
        for bot in self.bots.values():
            # 1. 感知（视线 + 听觉 + 记忆 + 可疑度）
            self.perception_system.update(bot, self.player_pos, self.time, dt,
                                          noise_events=self.noise_events)
            # 2. 通信：接收队友共享的情报
            for msg in self.bot_communication.receive(bot.bot_id, self.time):
                if msg.msg_type == "enemy_sighting":
                    bot.last_known_player_pos = tuple(msg.payload["pos"])
                    bot.last_seen_time = self.time
            # 3. 行为树：条件判断 + 决策
            context = self._build_context(bot)
            action = self.behavior_tree.tick(bot, context)
            # 4. 通信：发现敌人后广播给队友（带延迟）
            if context["player_visible"]:
                self.bot_communication.send(
                    bot.bot_id, "enemy_sighting", {"pos": self.player_pos}, self.time)
            self._history.append({
                "time": self.time, "bot_id": bot.bot_id, "action": action,
                "suspicion": bot.suspicion, "state": bot.state,
            })
        self.bot_communication.purge(self.time)
        self.noise_events = []

    def reset(self):
        self._state = {}
        self._history = []
        self.bots = {}
        self.bot_communication = BotCommunication()
        self.player_pos = None
        self.noise_events = []
        self.time = 0.0
