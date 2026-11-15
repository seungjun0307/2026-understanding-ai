import pygame, sys
pygame.init()

# NOTE: The following implementation is a sample and will not run without 
# the necessary model files (e.g., SAV files).
## ---------------- FILL HERE ---------------------------------
import pickle
import numpy as np

# 10개의 가능한 스트레이트 숫자 집합 사전 정의 (Ace=1 로우 및 Ace=14 하이 모두 포함)
_STRAIGHT_WINDOWS = [set(range(s, s + 5)) for s in range(1, 10)] + [{1, 10, 11, 12, 13}]
_ROYAL_SET = {1, 10, 11, 12, 13}


def extract_features_from_hand(hand):
    """
    3장 또는 4장의 카드 [(suit, rank), ...] 입력으로부터
    순서 불변(Permutation-Invariant) 족보 핵심 피처 벡터를 추출합니다.
    """
    k = len(hand)
    suits = [int(s) for s, _ in hand]
    ranks = sorted([int(r) for _, r in hand])

    # 1. 숫자(Rank) 빈도 및 페어/트리플/포카드 통계
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    freq_sorted = sorted(rank_counts.values(), reverse=True)

    max_freq = freq_sorted[0]
    second_freq = freq_sorted[1] if len(freq_sorted) > 1 else 0
    num_unique_ranks = len(freq_sorted)
    num_pairs = sum(1 for c in freq_sorted if c >= 2)

    # 현재 카드만으로 이미 확정된 최소 족보 점수
    if max_freq == 4:
        current_score = 7
    elif max_freq == 3:
        current_score = 3
    elif max_freq == 2 and second_freq == 2:
        current_score = 2
    elif max_freq == 2:
        current_score = 1
    else:
        current_score = 0

    # 2. 문양(Suit) 일치도 및 플러시(Flush) 드로우 가능 여부
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suit = max(suit_counts.values())
    # 5장 플러시가 되려면 현재 공개된 3장(또는 4장)의 문양이 모두 같아야만 가능함
    is_all_same_suit = 1 if max_suit == k else 0

    # 3. 스트레이트(Straight) 간격 및 가능 구간(Window) 개수 계산
    span_low = ranks[-1] - ranks[0]
    ranks_high = sorted([14 if r == 1 else r for r in ranks])
    span_high = ranks_high[-1] - ranks_high[0]
    min_span = min(span_low, span_high)

    straight_windows = 0
    rank_set = set(ranks)
    if num_unique_ranks == k:
        for w in _STRAIGHT_WINDOWS:
            if rank_set.issubset(w):
                straight_windows += 1

    is_sf_draw = 1 if (is_all_same_suit and straight_windows > 0) else 0
    is_royal_draw = 1 if (is_all_same_suit and num_unique_ranks == k and rank_set.issubset(_ROYAL_SET)) else 0

    # 정렬된 카드 숫자(k개) + 도메인 요약 피처(13개) 반환
    return ranks + [
        max_freq,
        second_freq,
        num_unique_ranks,
        num_pairs,
        current_score,
        max_suit,
        is_all_same_suit,
        span_low,
        span_high,
        min_span,
        straight_windows,
        is_sf_draw,
        is_royal_draw,
    ]


# 학습된 3장/4장 전용 분류 모델 로드
with open('poker-model3.sav', 'rb') as _f3:
    model3 = pickle.load(_f3)

with open('poker-model4.sav', 'rb') as _f4:
    model4 = pickle.load(_f4)

# 동일 입력 상태의 예측 결과를 재사용하기 위한 메모이제이션 캐시
_DECISION_CACHE = {}


# given parameters:
#   hand: list of tuples (suit, rank); length is 3 or 4
#   opposite_score: the score of opposite player
#   
# return:
#   True if you want to call,
#   False if you want to fold
def predict_call(hand, opposite_score):
    oppo = int(opposite_score)

    # [전략 1] 상대 점수가 0점(Nothing)이면 패배 가능성이 없으므로 무조건 CALL
    if oppo <= 0:
        return True

    feats = extract_features_from_hand(hand)
    k = len(hand)

    # 정렬된 카드 숫자(k개) 뒤에 위치한 주요 도메인 피처 추출
    current_score = feats[k + 4]
    is_all_same_suit = feats[k + 6]
    straight_windows = feats[k + 10]

    # [전략 2] 현재 확보한 최소 족보 점수가 상대 점수 이상이면 패배하지 않으므로 즉시 CALL
    if current_score >= oppo:
        return True

    # [전략 3] 족보 달성 가능성과 기대 수익(EV)을 고려한 조기 FOLD 판단
    if k == 4:
        # 4장 노페어 + 무늬 불일치 + 스트레이트 불가 시 최대 1점(상대 1점 기준 EV=-2.25 < -2.0)
        if current_score == 0 and is_all_same_suit == 0 and straight_windows == 0:
            return False
        # 4장 원페어(1점)면 마지막 1장으로 도달 가능한 최대 점수는 3점(트리플)
        if current_score == 1 and oppo >= 4:
            return False
        # 4장 투페어(2점)면 마지막 1장으로 최대 6점(풀하우스, 확률 4/48 -> 상대 3~5점 기준 EV=-2.50 < -2.0)
        if current_score == 2 and oppo >= 3:
            return False
        # 4장 트리플(3점)이면 마지막 1장으로 도달 가능한 최대 점수는 7점(포카드)
        if current_score == 3 and oppo >= 8:
            return False
    elif k == 3:
        # 3장 노페어 + 무늬 불일치 + 스트레이트 불가 패는 상대가 1점 이상일 때 기대 수익이 손절(-1.0)보다 낮으므로 즉시 FOLD
        if current_score == 0 and is_all_same_suit == 0 and straight_windows == 0:
            return False
        # 3장 원페어(1점)면 무늬가 이미 2개 이상 섞여 있어 최대 도달 가능 점수는 7점(포카드)
        if current_score == 1 and oppo >= 8:
            return False

    # [전략 4] 동일 상태 공간 캐시 조회
    cache_key = (k, tuple(feats), oppo)
    if cache_key in _DECISION_CACHE:
        return _DECISION_CACHE[cache_key]

    # [전략 5] Scikit-Learn 분류기 확률 예측 및 실제 현금($) 기댓값(EV) 계산
    X_input = np.asarray([feats], dtype=np.float32)

    if k == 3:
        probs = model3.predict_proba(X_input)[0]
        classes = model3.classes_
        threshold = model3.best_thresholds_.get(oppo, -1.0)
    else:
        probs = model4.predict_proba(X_input)[0]
        classes = model4.classes_
        threshold = model4.best_thresholds_.get(oppo, -2.0)

    p_win = 0.0
    p_lose = 0.0
    for cls, p in zip(classes, probs):
        if cls > oppo:
            p_win += float(p)
        elif cls < oppo:
            p_lose += float(p)

    # 승리 시 +$3, 무승부 시 $0, 패배 시 -$3 반영
    expected_money_change = 3.0 * (p_win - p_lose)
    decision = bool(expected_money_change >= threshold)

    _DECISION_CACHE[cache_key] = decision
    return decision
## -------------------------------------------------

import pandas as pd
cand_hands = pd.read_csv('poker-hand-testing.csv')

def pick_opposite_score():
    return cand_hands['Poker Hand'].sample().values[0]

def pick_five_cards():
    sam = cand_hands.sample().values[0]
    score = sam[-1]
    lis = sam[:-1]
    return (
        [ (a, b) for a, b in zip(lis[::2], lis[1::2]) ],
        score)


# game money: initially, $1000
money = 1000

# game count
#game_count = 1000

# initially, pick five cards
full_hand, hand_score = pick_five_cards()
print(full_hand, hand_score)

# currently, three flipped cards
flipped = 3
current_cards = full_hand[:flipped]

# current game's opposite hand score
opposite_hand_score = pick_opposite_score()

#game_count = game_count - 1
money = money - 1

font = pygame.font.SysFont("comicsans",20)
SCREEN_WIDTH, SCREEN_HEIGHT = 512, 512

# create window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("POKER")

CARD_WIDTH, CARD_HEIGHT = 60, 100
SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT = 32, 64

# create a surface object, image is drawn on it.
imp = pygame.image.load("cards.jpg")
imp = pygame.transform.scale(imp, (SUIT_ICON_WIDTH*4, SUIT_ICON_HEIGHT))

# list of suit icons (0: hear, 1: spade, 2: diamond, 3: club)
suits = list()

# 0: heart
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH*3, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 1: spade
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (0, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 2: diamond
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 3: club
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH*2, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# colors
white = (255, 255, 255)
black = (0, 0, 0)
gray = (128, 128, 128)

def draw_card(suit, rank, pos):
    # white card
    rect = pygame.Rect( pos, (CARD_WIDTH, CARD_HEIGHT) )
    pygame.draw.rect(screen, white, rect)

    # suit icon
    screen.blit(
        suits[suit], 
        (pos[0] + CARD_WIDTH/2 - SUIT_ICON_WIDTH/2, 
            pos[1] + CARD_HEIGHT*2/3 - SUIT_ICON_HEIGHT/2))

    # rank
    rank_text = font.render("{}".format(rank), True, black)
    rank_rect = rank_text.get_rect()
    rank_rect.center = (pos[0] + CARD_WIDTH/2, pos[1] + CARD_HEIGHT/4)
    screen.blit(rank_text, rank_rect)

def gamemoney_text():
    text = font.render("Your money: ${}".format(money), True, white)
    rank_rect = text.get_rect()
    rank_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/6)
    screen.blit(text, rank_rect)

def opposite_hand_text():
    text = font.render(
        "The opposite's hand score: {}".format(opposite_hand_score), 
        True, white)
    rank_rect = text.get_rect()
    rank_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/6 + rank_rect.height)
    screen.blit(text, rank_rect)

BUTTON_WIDTH, BUTTON_HEIGHT = 120, 50

def draw_button(center_pos, button_text, button_color):
    rect = pygame.Rect((0, 0, BUTTON_WIDTH, BUTTON_HEIGHT))
    rect.center = center_pos
    pygame.draw.rect(screen, button_color, rect) 

    # button text
    text = font.render(button_text, True, black)
    text_rect = text.get_rect()
    text_rect.center = rect.center
    screen.blit(text, text_rect)

msg = None
MESSAGE_HEIGHT = 40
def display_message(_msg):
    if _msg is not None:
        # white message board at the top of screen
        rect = pygame.Rect( (0, 0), (SCREEN_WIDTH, MESSAGE_HEIGHT) )
        pygame.draw.rect(screen, white, rect)
        
        # display message text
        text = font.render(_msg, True, black)
        rect = text.get_rect()
        rect.center = (SCREEN_WIDTH/2, MESSAGE_HEIGHT/2)
        screen.blit(text, rect)

# bet and die buttons at the bottom of screen
BET_BUTTON_CENTER = (SCREEN_WIDTH/4, SCREEN_HEIGHT*4/5)
bet_button_color = white

def draw_bet_button():
    draw_button(BET_BUTTON_CENTER, "CALL", bet_button_color)

# answer if pos is on the button
def on_bet_button(pos):
    return (abs(pos[0] - BET_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - BET_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

DIE_BUTTON_CENTER = (SCREEN_WIDTH/2, SCREEN_HEIGHT*4/5)
die_button_color = white

def draw_die_button():
    draw_button(DIE_BUTTON_CENTER, "FOLD", die_button_color)

def on_die_button(pos):
    return (abs(pos[0] - DIE_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - DIE_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

# auto play with classifier
CLS_BUTTON_CENTER = (SCREEN_WIDTH*3/4, SCREEN_HEIGHT*4/5)
cls_button_color = white

def draw_cls_button():
    draw_button(CLS_BUTTON_CENTER, "AUTO", cls_button_color)

def on_cls_button(pos):
    return (abs(pos[0] - CLS_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - CLS_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

# play sound when cards are revealed
flip_sound = pygame.mixer.Sound("flipcard.wav")
def play_cardflip_sound(rep):
    pygame.mixer.Sound.play(flip_sound, loops=rep-1)

def draw_current_cards():
    for i, (suit, rank) in enumerate(current_cards):
        draw_card(
            suit-1, rank, # suit and rank 
            ((i+1)*SCREEN_WIDTH/7, SCREEN_HEIGHT*2/5))

# game state for updating screen
# 0: beginning of game - three cards are flipped
# 1: bet - one more card is flipped
# 2: bet & game over - the last card is flipped and
#    win/tie/lose messages are shown
#    then, go to the next game after 5 secs
# 3: die & game over - message shown then,
#    go to the next game after 5 secs
# 9: wait for the next action
update_state = 0

def draw_items_on_screen():
    screen.fill(black)
    opposite_hand_text()
    gamemoney_text()
    draw_bet_button()
    draw_die_button()
    draw_cls_button()
    draw_current_cards()

    if msg is not None:
        display_message(msg)

    pygame.display.flip()
    pygame.time.delay(200)

# drawing cards and playing sound
def update_cards_and_sounds():
    global flipped, update_state, full_hand, hand_score,\
        money, msg, current_cards, opposite_hand_score

    if update_state == 0:
        # at the beginning of game, three cards are flipped
        play_cardflip_sound(3)
        draw_items_on_screen()

        # then, the screen of waiting the next action
        update_state = 9

    elif update_state == 1:
        # check button clicked and now four cards revealed
        play_cardflip_sound(1)
        draw_items_on_screen()

        # then, the screen of waiting the next action
        update_state = 9

    elif update_state == 2 or update_state == 3:
        # the last cards are revealed
        play_cardflip_sound(1)
        draw_items_on_screen()

        # delay another 1 sec for showing message
        pygame.time.delay(1000)
        msg = None
        draw_items_on_screen()

        # a new poker hand is drawn
        full_hand, hand_score = pick_five_cards()
        print(full_hand, hand_score)

        # three flipped cards
        flipped = 3
        current_cards = full_hand[:flipped]

        # opposite's hand score
        opposite_hand_score = pick_opposite_score()
        money = money - 1
        update_state = 0


game_stop = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and not game_stop:
            click_pos = pygame.mouse.get_pos()

            auto_bet = False
            auto_die = False
            if on_cls_button(click_pos):
                if predict_call(current_cards, opposite_hand_score):
                    print("auto call!")
                    auto_bet = True
                    display_message("AI calls")
                    pygame.display.update()
                    pygame.time.delay(800)
                else:
                    print("auto fold!")
                    auto_die = True
                    display_message("AI folds")
                    pygame.display.update()
                    pygame.time.delay(800)

            if on_bet_button(click_pos) or auto_bet:
                flipped = flipped + 1
                current_cards = full_hand[:flipped]
                money = money - 1

                # if game over,
                if flipped == 5:
                    # case 1: win
                    if hand_score > opposite_hand_score:
                        money = money + 6
                        msg = "Win"
                        update_state = 2
                    # case 2: tie
                    elif hand_score == opposite_hand_score:
                        money = money + 3
                        msg = "Tie"
                        update_state = 2
                    # case 3: lose
                    else:
                        msg = "Lose"
                        update_state = 2
                # if cards can be flipped more,
                else:
                    update_state = 1
                        
            elif on_die_button(click_pos) or auto_die:
                msg = "Fold, go on to the next"
                update_state = 3

    # show cards and info message including sound
    update_cards_and_sounds()

    pygame.time.delay(200)