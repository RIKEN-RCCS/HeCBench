// bonds-kokkos/main.cpp
// Kokkos port of the bonds benchmark
// Original by Scott Grauer-Gray; Kokkos port adapted from bonds-omp

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cstring>
#include <cmath>
#include <Kokkos_Core.hpp>
#include "../bonds-omp/bondsStructs.h"

// ============================================================
// Forward declarations for all device helper functions
// ============================================================

KOKKOS_INLINE_FUNCTION int monthLengthKernelGpu(int month, bool leapYear);
KOKKOS_INLINE_FUNCTION int monthOffsetKernelGpu(int m, bool leapYear);
KOKKOS_INLINE_FUNCTION int yearOffsetKernelGpu(int y);
KOKKOS_INLINE_FUNCTION bool isLeapKernelGpu(int y);
KOKKOS_INLINE_FUNCTION bondsDateStruct intializeDateKernelGpu(int d, int m, int y);
KOKKOS_INLINE_FUNCTION int dayCountGpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter);
KOKKOS_INLINE_FUNCTION dataType yearFractionGpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter);
KOKKOS_INLINE_FUNCTION dataType couponNotionalGpu();
KOKKOS_INLINE_FUNCTION dataType bondNotionalGpu();
KOKKOS_INLINE_FUNCTION dataType fixedRateCouponNominalGpu();
KOKKOS_INLINE_FUNCTION bool eventHasOccurredGpu(bondsDateStruct currDate, bondsDateStruct eventDate);
KOKKOS_INLINE_FUNCTION bool cashFlowHasOccurredGpu(bondsDateStruct refDate, bondsDateStruct eventDate);
KOKKOS_INLINE_FUNCTION bondsDateStruct advanceDateGpu(bondsDateStruct date, int numMonthsAdvance);
KOKKOS_INLINE_FUNCTION dataType getDirtyPriceGpu(bondStruct *bond,
    bondsYieldTermStruct *discountCurve,
    bondsDateStruct *currDate,
    int bondNum, cashFlowsStruct cashFlows, int numLegs);
KOKKOS_INLINE_FUNCTION dataType getAccruedAmountGpu(bondsDateStruct *maturityDate,
    bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs);
KOKKOS_INLINE_FUNCTION dataType bondFunctionsAccruedAmountGpu(bondsDateStruct *maturityDate,
    bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs);
KOKKOS_INLINE_FUNCTION dataType cashFlowsAccruedAmountGpu(cashFlowsStruct cashFlows,
    bool includecurrDateFlows, bondsDateStruct currDate,
    int numLegs, bondsDateStruct *maturityDate, int bondNum);
KOKKOS_INLINE_FUNCTION dataType fixedRateCouponAccruedAmountGpu(cashFlowsStruct cashFlows,
    int numLeg, bondsDateStruct d, bondsDateStruct *maturityDate, int bondNum);
KOKKOS_INLINE_FUNCTION dataType cashFlowsNpvGpu(cashFlowsStruct cashFlows,
    bondsYieldTermStruct discountCurve, bool includecurrDateFlows,
    bondsDateStruct currDate, bondsDateStruct npvDate, int numLegs);
KOKKOS_INLINE_FUNCTION dataType bondsYieldTermStructureDiscountGpu(bondsYieldTermStruct ytStruct,
    bondsDateStruct t);
KOKKOS_INLINE_FUNCTION dataType flatForwardDiscountImplGpu(intRateStruct intRate, dataType t);
KOKKOS_INLINE_FUNCTION dataType interestRateDiscountFactorGpu(intRateStruct intRate, dataType t);
KOKKOS_INLINE_FUNCTION dataType interestRateCompoundFactorGpuTwoArgs(intRateStruct intRate, dataType t);
KOKKOS_INLINE_FUNCTION dataType fixedRateCouponAmountGpu(cashFlowsStruct cashFlows, int numLeg);
KOKKOS_INLINE_FUNCTION dataType interestRateCompoundFactorGpu(intRateStruct intRate,
    bondsDateStruct d1, bondsDateStruct d2, int dayCounter);
KOKKOS_INLINE_FUNCTION dataType interestRateImpliedRateGpu(dataType compound,
    int comp, dataType freq, dataType t);
KOKKOS_INLINE_FUNCTION int cashFlowsNextCashFlowNumGpu(cashFlowsStruct cashFlows,
    bondsDateStruct currDate, int numLegs);
KOKKOS_INLINE_FUNCTION dataType getBondYieldGpu(dataType cleanPrice, int dc, int comp,
    dataType freq, bondsDateStruct settlement, dataType accuracy, int maxEvaluations,
    bondStruct *bond, bondsDateStruct *maturityDate,
    int bondNum, cashFlowsStruct cashFlows, int numLegs);
KOKKOS_INLINE_FUNCTION dataType getBondFunctionsYieldGpu(dataType cleanPrice, int dc,
    int comp, dataType freq, bondsDateStruct settlement, dataType accuracy,
    int maxEvaluations, bondsDateStruct *maturityDate,
    int bondNum, cashFlowsStruct cashFlows, int numLegs);
KOKKOS_INLINE_FUNCTION dataType getCashFlowsYieldGpu(cashFlowsStruct leg, dataType npv,
    int dayCounter, int compounding, dataType frequency, bool includecurrDateFlows,
    bondsDateStruct currDate, bondsDateStruct npvDate, int numLegs,
    dataType accuracy, int maxIterations, dataType guess);
KOKKOS_INLINE_FUNCTION dataType solverSolveGpu(solverStruct solver, irrFinderStruct f,
    dataType accuracy, dataType guess, dataType step,
    cashFlowsStruct cashFlows, int numLegs);
KOKKOS_INLINE_FUNCTION dataType cashFlowsNpvYieldGpu(cashFlowsStruct cashFlows,
    intRateStruct y, bool includecurrDateFlows,
    bondsDateStruct currDate, bondsDateStruct npvDate, int numLegs);
KOKKOS_INLINE_FUNCTION dataType fOpGpu(irrFinderStruct f, dataType y,
    cashFlowsStruct cashFlows, int numLegs);
KOKKOS_INLINE_FUNCTION dataType fDerivativeGpu(irrFinderStruct f, dataType y,
    cashFlowsStruct cashFlows, int numLegs);
KOKKOS_INLINE_FUNCTION bool closeGpu(dataType x, dataType y);
KOKKOS_INLINE_FUNCTION bool closeGpuThreeArgs(dataType x, dataType y, int n);
KOKKOS_INLINE_FUNCTION dataType solveImplGpu(solverStruct solver, irrFinderStruct f,
    dataType xAccuracy, cashFlowsStruct cashFlows, int numLegs);
KOKKOS_INLINE_FUNCTION dataType modifiedDurationGpu(cashFlowsStruct cashFlows,
    intRateStruct y, bool includecurrDateFlows,
    bondsDateStruct currDate, bondsDateStruct npvDate, int numLegs);

// ============================================================
// Device function definitions (ported from bondsKernelsGpu.cpp)
// ============================================================

KOKKOS_INLINE_FUNCTION
int monthLengthKernelGpu(int month, bool leapYear)
{
  int MonthLength[12];
  MonthLength[0]=31;
  MonthLength[1]=28;
  MonthLength[2]=31;
  MonthLength[3]=30;
  MonthLength[4]=31;
  MonthLength[5]=30;
  MonthLength[6]=31;
  MonthLength[7]=31;
  MonthLength[8]=30;
  MonthLength[9]=31;
  MonthLength[10]=30;
  MonthLength[11]=31;

  int MonthLeapLength[12];
  MonthLeapLength[0]=31;
  MonthLeapLength[1]=29;
  MonthLeapLength[2]=31;
  MonthLeapLength[3]=30;
  MonthLeapLength[4]=31;
  MonthLeapLength[5]=30;
  MonthLeapLength[6]=31;
  MonthLeapLength[7]=31;
  MonthLeapLength[8]=30;
  MonthLeapLength[9]=31;
  MonthLeapLength[10]=30;
  MonthLeapLength[11]=31;

  return (leapYear? MonthLeapLength[month-1] : MonthLength[month-1]);
}


KOKKOS_INLINE_FUNCTION
int monthOffsetKernelGpu(int m, bool leapYear)
{
  int MonthOffset[13];
  MonthOffset[0]=0;
  MonthOffset[1]=31;
  MonthOffset[2]=59;
  MonthOffset[3]=90;
  MonthOffset[4]=120;
  MonthOffset[5]=151;
  MonthOffset[6]=181;
  MonthOffset[7]=212;
  MonthOffset[8]=243;
  MonthOffset[9]=273;
  MonthOffset[10]=304;
  MonthOffset[11]=334;
  MonthOffset[12]=365;

  int MonthLeapOffset[13];
  MonthLeapOffset[0]=0;
  MonthLeapOffset[1]=31;
  MonthLeapOffset[2]=60;
  MonthLeapOffset[3]=91;
  MonthLeapOffset[4]=121;
  MonthLeapOffset[5]=152;
  MonthLeapOffset[6]=182;
  MonthLeapOffset[7]=213;
  MonthLeapOffset[8]=244;
  MonthLeapOffset[9]=274;
  MonthLeapOffset[10]=305;
  MonthLeapOffset[11]=335;
  MonthLeapOffset[12]=366;

  return (leapYear? MonthLeapOffset[m-1] : MonthOffset[m-1]);
}


KOKKOS_INLINE_FUNCTION
int yearOffsetKernelGpu(int y)
{
  int YearOffset[121];
  YearOffset[0] = 0;;
  YearOffset[1] = 366;;
  YearOffset[2] = 731;
  YearOffset[3] = 1096;
  YearOffset[4] = 1461;
  YearOffset[5] = 1827;
  YearOffset[6] = 2192;
  YearOffset[7] = 2557;
  YearOffset[8] = 2922;
  YearOffset[9] = 3288;
  YearOffset[10] = 3653;
  YearOffset[11] = 4018;
  YearOffset[12] = 4383;
  YearOffset[13] = 4749;
  YearOffset[14] = 5114;
  YearOffset[15] = 5479;
  YearOffset[16] = 5844;
  YearOffset[17] = 6210;
  YearOffset[18] = 6575;
  YearOffset[19] = 6940;
  YearOffset[20] = 7305;
  YearOffset[21] = 7671;
  YearOffset[22] = 8036;
  YearOffset[23] = 8401;
  YearOffset[24] = 8766;
  YearOffset[25] = 9132;
  YearOffset[26] = 9497;
  YearOffset[27] = 9862;
  YearOffset[28] = 10227;
  YearOffset[29] = 10593;
  YearOffset[30] = 10958;
  YearOffset[31] = 11323;
  YearOffset[32] = 11688;
  YearOffset[33] = 12054;
  YearOffset[34] = 12419;
  YearOffset[35] = 12784;
  YearOffset[36] = 13149;
  YearOffset[37] = 13515;
  YearOffset[38] = 13880;
  YearOffset[39] = 14245;
  YearOffset[40] = 14610;
  YearOffset[41] = 14976;
  YearOffset[42] = 15341;
  YearOffset[43] = 15706;
  YearOffset[44] = 16071;
  YearOffset[45] = 16437;
  YearOffset[46] = 16802;
  YearOffset[47] = 17167;
  YearOffset[48] = 17532;
  YearOffset[49] = 17898;
  YearOffset[50] = 18263;
  YearOffset[51] = 18628;
  YearOffset[52] = 18993;
  YearOffset[53] = 19359;
  YearOffset[54] = 19724;
  YearOffset[55] = 20089;
  YearOffset[56] = 20454;
  YearOffset[57] = 20820;
  YearOffset[58] = 21185;
  YearOffset[59] = 21550;
  YearOffset[60] = 21915;
  YearOffset[61] = 22281;
  YearOffset[62] = 22646;
  YearOffset[63] = 23011;
  YearOffset[64] = 23376;
  YearOffset[65] = 23742;
  YearOffset[66] = 24107;
  YearOffset[67] = 24472;
  YearOffset[68] = 24837;
  YearOffset[69] = 25203;
  YearOffset[70] = 25568;
  YearOffset[71] = 25933;
  YearOffset[72] = 26298;
  YearOffset[73] = 26664;
  YearOffset[74] = 27029;
  YearOffset[75] = 27394;
  YearOffset[76] = 27759;
  YearOffset[77] = 28125;
  YearOffset[78] = 28490;
  YearOffset[79] = 28855;
  YearOffset[80] = 29220;
  YearOffset[81] = 29586;
  YearOffset[82] = 29951;
  YearOffset[83] = 30316;
  YearOffset[84] = 30681;
  YearOffset[85] = 31047;
  YearOffset[86] = 31412;
  YearOffset[87] = 31777;
  YearOffset[88] = 32142;
  YearOffset[89] = 32508;
  YearOffset[90] = 32873;
  YearOffset[91] = 33238;
  YearOffset[92] = 33603;
  YearOffset[93] = 33969;
  YearOffset[94] = 34334;
  YearOffset[95] = 34699;
  YearOffset[96] = 35064;
  YearOffset[97] = 35430;
  YearOffset[98] = 35795;
  YearOffset[99] = 36160;
  YearOffset[100] = 36525;
  YearOffset[101] = 36891;
  YearOffset[102] = 37256;
  YearOffset[103] = 37621;
  YearOffset[104] = 37986;
  YearOffset[105] = 38352;
  YearOffset[106] = 38717;
  YearOffset[107] = 39082;
  YearOffset[108] = 39447;
  YearOffset[109] = 39813;
  YearOffset[110] = 40178;
  YearOffset[111] = 40543;
  YearOffset[112] = 40908;
  YearOffset[113] = 41274;
  YearOffset[114] = 41639;
  YearOffset[115] = 42004;
  YearOffset[116] = 42369;
  YearOffset[117] = 42735;
  YearOffset[118] = 43100;
  YearOffset[119] = 42735;
  YearOffset[120] = 43830;

  return YearOffset[y-1900];
}


KOKKOS_INLINE_FUNCTION
bool isLeapKernelGpu(int y)
{
  bool YearIsLeap[121];

  YearIsLeap[0] = 1;;
  YearIsLeap[1] = 0;;
  YearIsLeap[2] = 0;
  YearIsLeap[3] = 0;
  YearIsLeap[4] = 1;
  YearIsLeap[5] = 0;
  YearIsLeap[6] = 0;
  YearIsLeap[7] = 0;
  YearIsLeap[8] = 1;
  YearIsLeap[9] = 0;
  YearIsLeap[10] = 0;
  YearIsLeap[11] = 0;
  YearIsLeap[12] = 1;
  YearIsLeap[13] = 0;
  YearIsLeap[14] = 0;
  YearIsLeap[15] = 0;
  YearIsLeap[16] = 1;
  YearIsLeap[17] = 0;
  YearIsLeap[18] = 0;
  YearIsLeap[19] = 0;
  YearIsLeap[20] = 1;
  YearIsLeap[21] = 0;
  YearIsLeap[22] = 0;
  YearIsLeap[23] = 0;
  YearIsLeap[24] = 1;
  YearIsLeap[25] = 0;
  YearIsLeap[26] = 0;
  YearIsLeap[27] = 0;
  YearIsLeap[28] = 1;
  YearIsLeap[29] = 0;
  YearIsLeap[30] = 0;
  YearIsLeap[31] = 0;
  YearIsLeap[32] = 1;
  YearIsLeap[33] = 0;
  YearIsLeap[34] = 0;
  YearIsLeap[35] = 0;
  YearIsLeap[36] = 1;
  YearIsLeap[37] = 0;
  YearIsLeap[38] = 0;
  YearIsLeap[39] = 0;
  YearIsLeap[40] = 1;
  YearIsLeap[41] = 0;
  YearIsLeap[42] = 0;
  YearIsLeap[43] = 0;
  YearIsLeap[44] = 1;
  YearIsLeap[45] = 0;
  YearIsLeap[46] = 0;
  YearIsLeap[47] = 0;
  YearIsLeap[48] = 1;
  YearIsLeap[49] = 0;
  YearIsLeap[50] = 0;
  YearIsLeap[51] = 0;
  YearIsLeap[52] = 1;
  YearIsLeap[53] = 0;
  YearIsLeap[54] = 0;
  YearIsLeap[55] = 0;
  YearIsLeap[56] = 1;
  YearIsLeap[57] = 0;
  YearIsLeap[58] = 0;
  YearIsLeap[59] = 0;
  YearIsLeap[60] = 1;
  YearIsLeap[61] = 0;
  YearIsLeap[62] = 0;
  YearIsLeap[63] = 0;
  YearIsLeap[64] = 1;
  YearIsLeap[65] = 0;
  YearIsLeap[66] = 0;
  YearIsLeap[67] = 0;
  YearIsLeap[68] = 1;
  YearIsLeap[69] = 0;
  YearIsLeap[70] = 0;
  YearIsLeap[71] = 0;
  YearIsLeap[72] = 1;
  YearIsLeap[73] = 0;
  YearIsLeap[74] = 0;
  YearIsLeap[75] = 0;
  YearIsLeap[76] = 1;
  YearIsLeap[77] = 0;
  YearIsLeap[78] = 0;
  YearIsLeap[79] = 0;
  YearIsLeap[80] = 1;
  YearIsLeap[81] = 0;
  YearIsLeap[82] = 0;
  YearIsLeap[83] = 0;
  YearIsLeap[84] = 1;
  YearIsLeap[85] = 0;
  YearIsLeap[86] = 0;
  YearIsLeap[87] = 0;
  YearIsLeap[88] = 1;
  YearIsLeap[89] = 0;
  YearIsLeap[90] = 0;
  YearIsLeap[91] = 0;
  YearIsLeap[92] = 1;
  YearIsLeap[93] = 0;
  YearIsLeap[94] = 0;
  YearIsLeap[95] = 0;
  YearIsLeap[96] = 1;
  YearIsLeap[97] = 0;
  YearIsLeap[98] = 0;
  YearIsLeap[99] = 0;
  YearIsLeap[100] = 1;
  YearIsLeap[101] = 0;
  YearIsLeap[102] = 0;
  YearIsLeap[103] = 0;
  YearIsLeap[104] = 1;
  YearIsLeap[105] = 0;
  YearIsLeap[106] = 0;
  YearIsLeap[107] = 0;
  YearIsLeap[108] = 1;
  YearIsLeap[109] = 0;
  YearIsLeap[110] = 0;
  YearIsLeap[111] = 0;
  YearIsLeap[112] = 1;
  YearIsLeap[113] = 0;
  YearIsLeap[114] = 0;
  YearIsLeap[115] = 0;
  YearIsLeap[116] = 1;
  YearIsLeap[117] = 0;
  YearIsLeap[118] = 0;
  YearIsLeap[119] = 0;
  YearIsLeap[120] = 1;

  return YearIsLeap[y-1900];
}


KOKKOS_INLINE_FUNCTION
bondsDateStruct intializeDateKernelGpu(int d, int m, int y)
{
  bondsDateStruct currDate;

  currDate.day = d;
  currDate.month = m;
  currDate.year = y;

  bool leap = isLeapKernelGpu(y);
  int offset = monthOffsetKernelGpu(m,leap);

  currDate.dateSerialNum = d + offset + yearOffsetKernelGpu(y);

  return currDate;
}


KOKKOS_INLINE_FUNCTION
dataType yearFractionGpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter)
{
  return dayCountGpu(d1, d2, dayCounter) / (dataType)360.0;
}


KOKKOS_INLINE_FUNCTION
int dayCountGpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter)
{
  if (dayCounter == USE_EXACT_DAY)
  {
    int dd1 = d1.day, dd2 = d2.day;
    int mm1 = d1.month, mm2 = d2.month;
    int yy1 = d1.year, yy2 = d2.year;

    if (dd2 == 31 && dd1 < 30)
    {
      dd2 = 1; mm2++;
    }

    return 360*(yy2-yy1) + 30*(mm2-mm1-1) + MAX(0, 30-dd1) + MIN(30, dd2);
  }
  else
  {
    return (d2.dateSerialNum - d1.dateSerialNum);
  }
}


KOKKOS_INLINE_FUNCTION
dataType couponNotionalGpu()
{
  return (dataType)100.0;
}

KOKKOS_INLINE_FUNCTION
dataType bondNotionalGpu()
{
  return (dataType)100.0;
}


KOKKOS_INLINE_FUNCTION
dataType fixedRateCouponNominalGpu()
{
  return (dataType)100.0;
}

KOKKOS_INLINE_FUNCTION
bool eventHasOccurredGpu(bondsDateStruct currDate, bondsDateStruct eventDate)
{
  return eventDate.dateSerialNum > currDate.dateSerialNum;
}


KOKKOS_INLINE_FUNCTION
bool cashFlowHasOccurredGpu(bondsDateStruct refDate, bondsDateStruct eventDate)
{
  return eventHasOccurredGpu(refDate, eventDate);
}


KOKKOS_INLINE_FUNCTION
bondsDateStruct advanceDateGpu(bondsDateStruct date, int numMonthsAdvance)
{
  int d = date.day;
  int m = date.month+numMonthsAdvance;
  int y = date.year;

  while (m > 12)
  {
    m -= 12;
    y += 1;
  }

  while (m < 1)
  {
    m += 12;
    y -= 1;
  }

  int length = monthLengthKernelGpu(m, isLeapKernelGpu(y));
  if (d > length)
    d = length;

  bondsDateStruct newDate = intializeDateKernelGpu(d, m, y);

  return newDate;
}


KOKKOS_INLINE_FUNCTION
dataType getDirtyPriceGpu(bondStruct *bond,
    bondsYieldTermStruct *discountCurve,
    bondsDateStruct *currDate,
    int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
  dataType currentNotional = bondNotionalGpu();
  bondsDateStruct currentDate = currDate[bondNum];

  if (currentDate.dateSerialNum < bond[bondNum].startDate.dateSerialNum)
  {
    currentDate = bond[bondNum].startDate;
  }

  return cashFlowsNpvGpu(cashFlows,
      discountCurve[bondNum], false, currentDate, currentDate, numLegs) * (dataType)100.0 / currentNotional;
}


KOKKOS_INLINE_FUNCTION
dataType getAccruedAmountGpu(bondsDateStruct *maturityDate, bondsDateStruct date,
    int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
  dataType currentNotional = bondNotionalGpu();
  if (currentNotional == (dataType)0.0)
    return (dataType)0.0;

  return cashFlowsAccruedAmountGpu(cashFlows, false, date, numLegs, maturityDate, bondNum) *
    (dataType)100.0 / bondNotionalGpu();
}


KOKKOS_INLINE_FUNCTION
dataType bondFunctionsAccruedAmountGpu(bondsDateStruct *maturityDate, bondsDateStruct date,
    int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
  return cashFlowsAccruedAmountGpu(cashFlows, false, date, numLegs, maturityDate, bondNum) *
    (dataType)100.0 / bondNotionalGpu();
}


KOKKOS_INLINE_FUNCTION
dataType cashFlowsAccruedAmountGpu(cashFlowsStruct cashFlows,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    int numLegs, bondsDateStruct *maturityDate, int bondNum)
{
  int legComputeNum = cashFlowsNextCashFlowNumGpu(cashFlows, currDate, numLegs);

  dataType result = 0.0;

  for (int i = legComputeNum; i < (numLegs); ++i)
  {
    result += fixedRateCouponAccruedAmountGpu(cashFlows, i, currDate, maturityDate, bondNum);
  }

  return result;
}


KOKKOS_INLINE_FUNCTION
dataType fixedRateCouponAccruedAmountGpu(cashFlowsStruct cashFlows, int numLeg,
    bondsDateStruct d, bondsDateStruct *maturityDate, int bondNum)
{
  if (d.dateSerialNum <= cashFlows.legs[numLeg].accrualStartDate.dateSerialNum ||
      d.dateSerialNum > maturityDate[bondNum].dateSerialNum)
  {
    return (dataType)0.0;
  }
  else
  {
    bondsDateStruct endDate = cashFlows.legs[numLeg].accrualEndDate;
    if (d.dateSerialNum < cashFlows.legs[numLeg].accrualEndDate.dateSerialNum)
    {
      endDate = d;
    }

    return fixedRateCouponNominalGpu()*(interestRateCompoundFactorGpu(cashFlows.intRate,
          cashFlows.legs[numLeg].accrualStartDate, endDate, cashFlows.dayCounter) - (dataType)1.0);
  }
}


KOKKOS_INLINE_FUNCTION
dataType cashFlowsNpvGpu(cashFlowsStruct cashFlows,
    bondsYieldTermStruct discountCurve,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    bondsDateStruct npvDate,
    int numLegs)
{
  npvDate = currDate;

  dataType totalNPV = 0.0;

  int i;

  for (i=0; i<numLegs; ++i) {
    if (!(cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate)))
      totalNPV += fixedRateCouponAmountGpu(cashFlows, i) *
        bondsYieldTermStructureDiscountGpu(discountCurve, cashFlows.legs[i].paymentDate);
  }

  return totalNPV/bondsYieldTermStructureDiscountGpu(discountCurve, npvDate);
}


KOKKOS_INLINE_FUNCTION
dataType bondsYieldTermStructureDiscountGpu(bondsYieldTermStruct ytStruct, bondsDateStruct t)
{
  ytStruct.intRate.rate = ytStruct.forward;
  ytStruct.intRate.freq = ytStruct.frequency;
  ytStruct.intRate.comp = ytStruct.compounding;
  return flatForwardDiscountImplGpu(ytStruct.intRate, yearFractionGpu(ytStruct.refDate, t, ytStruct.dayCounter));
}


KOKKOS_INLINE_FUNCTION
dataType flatForwardDiscountImplGpu(intRateStruct intRate, dataType t)
{
  return interestRateDiscountFactorGpu(intRate, t);
}


KOKKOS_INLINE_FUNCTION
dataType interestRateDiscountFactorGpu(intRateStruct intRate, dataType t)
{
  return (dataType)1.0/interestRateCompoundFactorGpuTwoArgs(intRate, t);
}


KOKKOS_INLINE_FUNCTION
dataType interestRateCompoundFactorGpuTwoArgs(intRateStruct intRate, dataType t)
{
  if (intRate.comp == SIMPLE_INTEREST)
    return (dataType)1.0 + intRate.rate*t;
  else if (intRate.comp == COMPOUNDED_INTEREST)
    return pow((dataType)1.0+intRate.rate/intRate.freq, intRate.freq*t);
  else if (intRate.comp == CONTINUOUS_INTEREST)
    return exp(intRate.rate*t);
  return (dataType)0.0;
}


KOKKOS_INLINE_FUNCTION
dataType fixedRateCouponAmountGpu(cashFlowsStruct cashFlows, int numLeg)
{
  if (cashFlows.legs[numLeg].amount == COMPUTE_AMOUNT)
  {
    return fixedRateCouponNominalGpu()*(interestRateCompoundFactorGpu(cashFlows.intRate,
          cashFlows.legs[numLeg].accrualStartDate,
          cashFlows.legs[numLeg].accrualEndDate, cashFlows.dayCounter) - (dataType)1.0);
  }
  else
  {
    return cashFlows.legs[numLeg].amount;
  }
}

KOKKOS_INLINE_FUNCTION
dataType interestRateCompoundFactorGpu(intRateStruct intRate, bondsDateStruct d1,
    bondsDateStruct d2, int dayCounter)
{
  dataType t = yearFractionGpu(d1, d2, dayCounter);
  return interestRateCompoundFactorGpuTwoArgs(intRate, t);
}


KOKKOS_INLINE_FUNCTION
dataType interestRateImpliedRateGpu(dataType compound, int comp, dataType freq, dataType t)
{
  dataType r = 0.0f;
  if (compound==(dataType)1.0)
  {
    r = 0.0;
  }
  else
  {
    switch (comp)
    {
      case SIMPLE_INTEREST:
        r = (compound - (dataType)1.0)/t;
        break;
      case COMPOUNDED_INTEREST:
        r = (pow((dataType)compound, (dataType)1.0/((freq)*t))-(dataType)1.0)*(freq);
        break;
    }
  }

  return r;
}


KOKKOS_INLINE_FUNCTION
int cashFlowsNextCashFlowNumGpu(cashFlowsStruct cashFlows,
    bondsDateStruct currDate,
    int numLegs)
{
  int i;
  for (i = 0; i < numLegs; ++i)
  {
    if ( ! (cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate) ))
      return i;
  }

  return (numLegs-1);
}


KOKKOS_INLINE_FUNCTION
dataType getBondYieldGpu(dataType cleanPrice,
    int dc,
    int comp,
    dataType freq,
    bondsDateStruct settlement,
    dataType accuracy,
    int maxEvaluations,
    bondStruct *bond,
    bondsDateStruct *maturityDate,
    int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
  dataType currentNotional = bondNotionalGpu();

  if (currentNotional == (dataType)0.0)
    return (dataType)0.0;

  if (bond[bondNum].startDate.dateSerialNum > settlement.dateSerialNum)
  {
    settlement = bond[bondNum].startDate;
  }

  return getBondFunctionsYieldGpu(cleanPrice, dc, comp, freq,
      settlement, accuracy, maxEvaluations,
      maturityDate, bondNum, cashFlows, numLegs);
}


KOKKOS_INLINE_FUNCTION
dataType getBondFunctionsYieldGpu(dataType cleanPrice,
    int dc,
    int comp,
    dataType freq,
    bondsDateStruct settlement,
    dataType accuracy,
    int maxEvaluations,
    bondsDateStruct *maturityDate,
    int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
  dataType dirtyPrice = cleanPrice + bondFunctionsAccruedAmountGpu(maturityDate, settlement, bondNum, cashFlows, numLegs);
  dirtyPrice /= (dataType)100.0 / bondNotionalGpu();

  return getCashFlowsYieldGpu(cashFlows, dirtyPrice,
      dc, comp, freq,
      false, settlement, settlement, numLegs,
      accuracy, maxEvaluations, (dataType)0.05);
}


KOKKOS_INLINE_FUNCTION
dataType getCashFlowsYieldGpu(cashFlowsStruct leg,
    dataType npv,
    int dayCounter,
    int compounding,
    dataType frequency,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    bondsDateStruct npvDate,
    int numLegs,
    dataType accuracy,
    int maxIterations,
    dataType guess)
{
  solverStruct solver;
  solver.maxEvaluations_ = maxIterations;
  irrFinderStruct objFunction;

  objFunction.npv = npv;
  objFunction.dayCounter = dayCounter;
  objFunction.comp = compounding;
  objFunction.freq = frequency;
  objFunction.includecurrDateFlows = includecurrDateFlows;
  objFunction.currDate = currDate;
  objFunction.npvDate = npvDate;

  return solverSolveGpu(solver, objFunction, accuracy, guess, guess/(dataType)10.0, leg, numLegs);
}


KOKKOS_INLINE_FUNCTION
dataType solverSolveGpu(solverStruct solver,
    irrFinderStruct f,
    dataType accuracy,
    dataType guess,
    dataType step,
    cashFlowsStruct cashFlows,
    int numLegs)
{
  accuracy = MAX(accuracy, QL_EPSILON_GPU);

  dataType growthFactor = (dataType)1.6;
  int flipflop = -1;

  solver.root_ = guess;
  solver.fxMax_ = fOpGpu(f, solver.root_, cashFlows, numLegs);

  if (closeGpu(solver.fxMax_,(dataType)0.0))
  {
    return solver.root_;
  }
  else if (closeGpu(solver.fxMax_, (dataType)0.0))
  {
    solver.xMin_ = (solver.root_ - step);
    solver.fxMin_ = fOpGpu(f, solver.xMin_, cashFlows, numLegs);
    solver.xMax_ = solver.root_;
  }
  else
  {
    solver.xMin_ = solver.root_;
    solver.fxMin_ = solver.fxMax_;
    solver.xMax_ = (solver.root_+step);
    solver.fxMax_ = fOpGpu(f, solver.xMax_, cashFlows, numLegs);
  }

  solver.evaluationNumber_ = 2;
  while (solver.evaluationNumber_ <= solver.maxEvaluations_)
  {
    if (solver.fxMin_*solver.fxMax_ <= (dataType)0.0)
    {
      if (closeGpu(solver.fxMin_, (dataType)0.0))
        return solver.xMin_;
      if (closeGpu(solver.fxMax_, (dataType)0.0))
        return solver.xMax_;
      solver.root_ = (solver.xMax_+solver.xMin_)/(dataType)2.0;
      return solveImplGpu(solver, f, accuracy, cashFlows, numLegs);
    }
    if (fabs(solver.fxMin_) < fabs(solver.fxMax_))
    {
      solver.xMin_ = (solver.xMin_+growthFactor*(solver.xMin_ - solver.xMax_));
      solver.fxMin_= fOpGpu(f, solver.xMin_, cashFlows, numLegs);
    }
    else if (fabs(solver.fxMin_) > fabs(solver.fxMax_))
    {
      solver.xMax_ = (solver.xMax_+growthFactor*(solver.xMax_ - solver.xMin_));
      solver.fxMax_= fOpGpu(f, solver.xMax_, cashFlows, numLegs);
    }
    else if (flipflop == -1)
    {
      solver.xMin_ = (solver.xMin_+growthFactor*(solver.xMin_ - solver.xMax_));
      solver.fxMin_= fOpGpu(f, solver.xMin_, cashFlows, numLegs);
      solver.evaluationNumber_++;
      flipflop = 1;
    }
    else if (flipflop == 1)
    {
      solver.xMax_ = (solver.xMax_+growthFactor*(solver.xMax_ - solver.xMin_));
      solver.fxMax_= fOpGpu(f, solver.xMax_, cashFlows, numLegs);
      flipflop = -1;
    }
    solver.evaluationNumber_++;
  }

  return (dataType)0.0;
}


KOKKOS_INLINE_FUNCTION
dataType cashFlowsNpvYieldGpu(cashFlowsStruct cashFlows,
    intRateStruct y,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    bondsDateStruct npvDate,
    int numLegs)
{
  dataType npv = 0.0;
  dataType discount = 1.0;
  bondsDateStruct lastDate;
  bool first = true;

  int i;
  for (i=0; i<numLegs; ++i)
  {
    if (cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate))
      continue;

    bondsDateStruct couponDate = cashFlows.legs[i].paymentDate;
    dataType amount = fixedRateCouponAmountGpu(cashFlows, i);
    if (first)
    {
      first = false;
      if (i > 0) {
        lastDate = advanceDateGpu(cashFlows.legs[i].paymentDate, -1*6);
      } else {
        lastDate = cashFlows.legs[i].accrualStartDate;
      }
      discount *= interestRateDiscountFactorGpu(y, yearFractionGpu(npvDate, couponDate, y.dayCounter));
    }
    else
    {
      discount *= interestRateDiscountFactorGpu(y, yearFractionGpu(lastDate, couponDate, y.dayCounter));
    }

    lastDate = couponDate;

    npv += amount * discount;
  }

  return npv;
}

KOKKOS_INLINE_FUNCTION
dataType fOpGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs)
{
  intRateStruct yield;

  yield.rate = y;
  yield.comp = f.comp;
  yield.freq = f.freq;
  yield.dayCounter = f.dayCounter;

  dataType NPV = cashFlowsNpvYieldGpu(cashFlows,
      yield,
      false,
      f.currDate,
      f.npvDate, numLegs);

  return (f.npv - NPV);
}


KOKKOS_INLINE_FUNCTION
dataType fDerivativeGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs)
{
  intRateStruct yield;
  yield.rate = y;
  yield.dayCounter = f.dayCounter;
  yield.comp = f.comp;
  yield.freq = f.freq;

  return modifiedDurationGpu(cashFlows, yield,
      f.includecurrDateFlows,
      f.currDate, f.npvDate, numLegs);
}


KOKKOS_INLINE_FUNCTION
bool closeGpu(dataType x, dataType y)
{
  return closeGpuThreeArgs(x,y,42);
}


KOKKOS_INLINE_FUNCTION
bool closeGpuThreeArgs(dataType x, dataType y, int n)
{
  dataType diff = fabs(x-y);
  dataType tolerance = n*QL_EPSILON_GPU;

  return diff <= tolerance*fabs(x) &&
    diff <= tolerance*fabs(y);
}

KOKKOS_INLINE_FUNCTION
dataType solveImplGpu(solverStruct solver, irrFinderStruct f,
    dataType xAccuracy, cashFlowsStruct cashFlows, int numLegs)
{
  dataType froot, dfroot, dx, dxold;
  dataType xh, xl;

  if (solver.fxMin_ < (dataType)0.0)
  {
    xl = solver.xMin_;
    xh = solver.xMax_;
  }
  else
  {
    xh = solver.xMin_;
    xl = solver.xMax_;
  }

  dxold = solver.xMax_ - solver.xMin_;
  dx = dxold;

  froot = fOpGpu(f, solver.root_, cashFlows, numLegs);
  dfroot = fDerivativeGpu(f, solver.root_, cashFlows, numLegs);

  ++solver.evaluationNumber_;

  while (solver.evaluationNumber_<=solver.maxEvaluations_)
  {
    if ((((solver.root_-xh)*dfroot-froot)*
          ((solver.root_-xl)*dfroot-froot) > (dataType)0.0)
        || (fabs((dataType)2.0*froot) > fabs(dxold*dfroot)))
    {
      dxold = dx;
      dx = (xh-xl)/(dataType)2.0;
      solver.root_=xl+dx;
    }
    else
    {
      dxold = dx;
      dx = froot/dfroot;
      solver.root_ -= dx;
    }

    if (fabs(dx) < xAccuracy)
      return solver.root_;
    froot = fOpGpu(f, solver.root_, cashFlows, numLegs);
    dfroot = fDerivativeGpu(f, solver.root_, cashFlows, numLegs);
    ++solver.evaluationNumber_;
    if (froot < (dataType)0.0)
      xl=solver.root_;
    else
      xh=solver.root_;
  }

  return solver.root_;
}


KOKKOS_INLINE_FUNCTION
dataType modifiedDurationGpu(cashFlowsStruct cashFlows,
    intRateStruct y,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    bondsDateStruct npvDate,
    int numLegs)
{
  dataType P = 0.0;
  dataType dPdy = 0.0;
  dataType r = y.rate;
  dataType N = y.freq;
  int dc = y.dayCounter;

  int i;
  for (i=0; i<numLegs; ++i)
  {
    if (!cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate))
    {
      dataType t = yearFractionGpu(npvDate,
          cashFlows.legs[i].paymentDate, dc);
      dataType c = fixedRateCouponAmountGpu(cashFlows, i);
      dataType B = interestRateDiscountFactorGpu(y, t);

      P += c * B;
      if (y.comp == SIMPLE_INTEREST)
        dPdy -= c * B*B * t;
      if (y.comp == COMPOUNDED_INTEREST)
        dPdy -= c * t * B/(1+r/N);
      if (y.comp == CONTINUOUS_INTEREST)
        dPdy -= c * B * t;
      if (y.comp == SIMPLE_THEN_COMPOUNDED_INTEREST)
      {
        if (t<=(dataType)1.0/N)
          dPdy -= c * B*B * t;
        else
          dPdy -= c * t * B/((dataType)1+r/N);
      }
    }
  }

  if (P == (dataType)0.0)
  {
    return (dataType)0.0;
  }
  return (-1*dPdy)/P;
}

// ============================================================
// Main Kokkos kernel function
// ============================================================

long getBondsResultsGpu(inArgsStruct inArgsHost, resultsStruct resultsFromGpu, int numBonds)
{
  // Allocate device Views
  Kokkos::View<bondsYieldTermStruct*> d_discountCurve("discountCurve", numBonds);
  Kokkos::View<bondsYieldTermStruct*> d_repoCurve("repoCurve", numBonds);
  Kokkos::View<bondsDateStruct*>      d_currDate("currDate", numBonds);
  Kokkos::View<bondsDateStruct*>      d_maturityDate("maturityDate", numBonds);
  Kokkos::View<dataType*>             d_bondCleanPrice("bondCleanPrice", numBonds);
  Kokkos::View<bondStruct*>           d_bond("bond", numBonds);
  Kokkos::View<dataType*>             d_dummyStrike("dummyStrike", numBonds);
  Kokkos::View<dataType*>             d_dirtyPrice("dirtyPrice", numBonds);
  Kokkos::View<dataType*>             d_accruedAmountCurrDate("accruedAmountCurrDate", numBonds);
  Kokkos::View<dataType*>             d_cleanPrice("cleanPrice", numBonds);
  Kokkos::View<dataType*>             d_bondForwardVal("bondForwardVal", numBonds);

  // Create host mirrors and copy input data from raw C arrays
  auto h_discountCurve  = Kokkos::create_mirror_view(d_discountCurve);
  auto h_repoCurve      = Kokkos::create_mirror_view(d_repoCurve);
  auto h_currDate       = Kokkos::create_mirror_view(d_currDate);
  auto h_maturityDate   = Kokkos::create_mirror_view(d_maturityDate);
  auto h_bondCleanPrice = Kokkos::create_mirror_view(d_bondCleanPrice);
  auto h_bond           = Kokkos::create_mirror_view(d_bond);
  auto h_dummyStrike    = Kokkos::create_mirror_view(d_dummyStrike);

  std::memcpy(h_discountCurve.data(),  inArgsHost.discountCurve,  numBonds * sizeof(bondsYieldTermStruct));
  std::memcpy(h_repoCurve.data(),      inArgsHost.repoCurve,      numBonds * sizeof(bondsYieldTermStruct));
  std::memcpy(h_currDate.data(),       inArgsHost.currDate,       numBonds * sizeof(bondsDateStruct));
  std::memcpy(h_maturityDate.data(),   inArgsHost.maturityDate,   numBonds * sizeof(bondsDateStruct));
  std::memcpy(h_bondCleanPrice.data(), inArgsHost.bondCleanPrice, numBonds * sizeof(dataType));
  std::memcpy(h_bond.data(),           inArgsHost.bond,           numBonds * sizeof(bondStruct));
  std::memcpy(h_dummyStrike.data(),    inArgsHost.dummyStrike,    numBonds * sizeof(dataType));

  Kokkos::deep_copy(d_discountCurve,  h_discountCurve);
  Kokkos::deep_copy(d_repoCurve,      h_repoCurve);
  Kokkos::deep_copy(d_currDate,       h_currDate);
  Kokkos::deep_copy(d_maturityDate,   h_maturityDate);
  Kokkos::deep_copy(d_bondCleanPrice, h_bondCleanPrice);
  Kokkos::deep_copy(d_bond,           h_bond);
  Kokkos::deep_copy(d_dummyStrike,    h_dummyStrike);

  Kokkos::Timer timer;

  Kokkos::parallel_for("bonds", numBonds, KOKKOS_LAMBDA(int bondNum) {
    bondStruct           *bond           = d_bond.data();
    bondsYieldTermStruct *discountCurve  = d_discountCurve.data();
    bondsDateStruct      *currDate       = d_currDate.data();
    bondsDateStruct      *maturityDate   = d_maturityDate.data();
    dataType             *bondCleanPrice = d_bondCleanPrice.data();
    dataType             *bondForwardVal = d_bondForwardVal.data();
    dataType             *dirtyPrice     = d_dirtyPrice.data();
    dataType             *accruedAmountCurrDate = d_accruedAmountCurrDate.data();
    dataType             *cleanPrice     = d_cleanPrice.data();

    int numLegs;

    int numCashFlows = 0;

    bondsDateStruct currCashflowDate = bond[bondNum].maturityDate;

    while (currCashflowDate.dateSerialNum > bond[bondNum].startDate.dateSerialNum)
    {
      numCashFlows++;
      currCashflowDate = advanceDateGpu(currCashflowDate, -6);
    }

    numLegs = numCashFlows+1;

    cashFlowsStruct cashFlows;
    couponStruct cashLegs[9];
    cashFlows.legs = cashLegs;

    cashFlows.intRate.dayCounter = USE_EXACT_DAY;
    cashFlows.intRate.rate  = bond[bondNum].rate;
    cashFlows.intRate.freq  = ANNUAL_FREQ;
    cashFlows.intRate.comp  = SIMPLE_INTEREST;
    cashFlows.dayCounter  = USE_EXACT_DAY;
    cashFlows.nominal  = (dataType)100.0;

    bondsDateStruct currStartDate = advanceDateGpu(bond[bondNum].maturityDate, (numLegs - 1)*-6);
    bondsDateStruct currEndDate = advanceDateGpu(currStartDate, 6);

    int cashFlowNum;
    for (cashFlowNum = 0; cashFlowNum < numLegs-1; cashFlowNum++)
    {
      cashFlows.legs[cashFlowNum].paymentDate = currEndDate;

      cashFlows.legs[cashFlowNum].accrualStartDate  = currStartDate;
      cashFlows.legs[cashFlowNum].accrualEndDate  = currEndDate;

      cashFlows.legs[cashFlowNum].amount = COMPUTE_AMOUNT;

      currStartDate = currEndDate;
      currEndDate = advanceDateGpu(currEndDate, 6);
    }

    cashFlows.legs[numLegs-1].paymentDate  = bond[bondNum].maturityDate;
    cashFlows.legs[numLegs-1].accrualStartDate = currDate[bondNum];
    cashFlows.legs[numLegs-1].accrualEndDate  = currDate[bondNum];
    cashFlows.legs[numLegs-1].amount = (dataType)100.0;

    bondForwardVal[bondNum] = getBondYieldGpu(bondCleanPrice[bondNum],
        USE_EXACT_DAY,
        COMPOUNDED_INTEREST,
        (dataType)2.0,
        currDate[bondNum],
        ACCURACY,
        100,
        bond,
        maturityDate,
        bondNum, cashFlows, numLegs);

    discountCurve[bondNum].forward = bondForwardVal[bondNum];
    dirtyPrice[bondNum] = getDirtyPriceGpu(bond, discountCurve, currDate, bondNum, cashFlows, numLegs);
    accruedAmountCurrDate[bondNum] = getAccruedAmountGpu(maturityDate, currDate[bondNum], bondNum, cashFlows, numLegs);

    cleanPrice[bondNum] = dirtyPrice[bondNum] - accruedAmountCurrDate[bondNum];
  });

  Kokkos::fence();
  long ktime = (long)(timer.seconds() * 1e6);

  // Copy results back to host
  auto h_dirtyPrice            = Kokkos::create_mirror_view(d_dirtyPrice);
  auto h_accruedAmountCurrDate = Kokkos::create_mirror_view(d_accruedAmountCurrDate);
  auto h_cleanPrice            = Kokkos::create_mirror_view(d_cleanPrice);
  auto h_bondForwardVal        = Kokkos::create_mirror_view(d_bondForwardVal);

  Kokkos::deep_copy(h_dirtyPrice,            d_dirtyPrice);
  Kokkos::deep_copy(h_accruedAmountCurrDate, d_accruedAmountCurrDate);
  Kokkos::deep_copy(h_cleanPrice,            d_cleanPrice);
  Kokkos::deep_copy(h_bondForwardVal,        d_bondForwardVal);

  std::memcpy(resultsFromGpu.dirtyPrice,            h_dirtyPrice.data(),            numBonds * sizeof(dataType));
  std::memcpy(resultsFromGpu.accruedAmountCurrDate, h_accruedAmountCurrDate.data(), numBonds * sizeof(dataType));
  std::memcpy(resultsFromGpu.cleanPrice,            h_cleanPrice.data(),            numBonds * sizeof(dataType));
  std::memcpy(resultsFromGpu.bondForwardVal,        h_bondForwardVal.data(),        numBonds * sizeof(dataType));

  return ktime;
}

// ============================================================
// CPU date helpers (from bondsEngine.cpp)
// ============================================================

int monthLengthCpu(int month, bool leapYear)
{
  int MonthLength[] = {
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
  };
  int MonthLeapLength[] = {
    31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
  };
  return (leapYear? MonthLeapLength[month-1] : MonthLength[month-1]);
}

int monthOffsetCpu(int m, bool leapYear)
{
  int MonthOffset[] = {
    0,  31,  59,  90, 120, 151,
    181, 212, 243, 273, 304, 334,
    365
  };
  int MonthLeapOffset[] = {
    0,  31,  60,  91, 121, 152,
    182, 213, 244, 274, 305, 335,
    366
  };
  return (leapYear? MonthLeapOffset[m-1] : MonthOffset[m-1]);
}

int yearOffsetCpu(int y)
{
  int YearOffset[] = {
    // 1900-1909
    0,  366,  731, 1096, 1461, 1827, 2192, 2557, 2922, 3288,
    // 1910-1919
    3653, 4018, 4383, 4749, 5114, 5479, 5844, 6210, 6575, 6940,
    // 1920-1929
    7305, 7671, 8036, 8401, 8766, 9132, 9497, 9862,10227,10593,
    // 1930-1939
    10958,11323,11688,12054,12419,12784,13149,13515,13880,14245,
    // 1940-1949
    14610,14976,15341,15706,16071,16437,16802,17167,17532,17898,
    // 1950-1959
    18263,18628,18993,19359,19724,20089,20454,20820,21185,21550,
    // 1960-1969
    21915,22281,22646,23011,23376,23742,24107,24472,24837,25203,
    // 1970-1979
    25568,25933,26298,26664,27029,27394,27759,28125,28490,28855,
    // 1980-1989
    29220,29586,29951,30316,30681,31047,31412,31777,32142,32508,
    // 1990-1999
    32873,33238,33603,33969,34334,34699,35064,35430,35795,36160,
    // 2000-2009
    36525,36891,37256,37621,37986,38352,38717,39082,39447,39813,
    // 2010-2019
    40178,40543,40908,41274,41639,42004,42369,42735,43100,43465,
    // 2020-2029
    43830,44196,44561,44926,45291,45657,46022,46387,46752,47118,
    // 2030-2039
    47483,47848,48213,48579,48944,49309,49674,50040,50405,50770,
    // 2040-2049
    51135,51501,51866,52231,52596,52962,53327,53692,54057,54423,
    // 2050-2059
    54788,55153,55518,55884,56249,56614,56979,57345,57710,58075,
    // 2060-2069
    58440,58806,59171,59536,59901,60267,60632,60997,61362,61728,
    // 2070-2079
    62093,62458,62823,63189,63554,63919,64284,64650,65015,65380,
    // 2080-2089
    65745,66111,66476,66841,67206,67572,67937,68302,68667,69033,
    // 2090-2099
    69398,69763,70128,70494,70859,71224,71589,71955,72320,72685,
    // 2100-2109
    73050,73415,73780,74145,74510,74876,75241,75606,75971,76337,
    // 2110-2119
    76702,77067,77432,77798,78163,78528,78893,79259,79624,79989,
    // 2120-2129
    80354,80720,81085,81450,81815,82181,82546,82911,83276,83642,
    // 2130-2139
    84007,84372,84737,85103,85468,85833,86198,86564,86929,87294,
    // 2140-2149
    87659,88025,88390,88755,89120,89486,89851,90216,90581,90947,
    // 2150-2159
    91312,91677,92042,92408,92773,93138,93503,93869,94234,94599,
    // 2160-2169
    94964,95330,95695,96060,96425,96791,97156,97521,97886,98252,
    // 2170-2179
    98617,98982,99347,99713,100078,100443,100808,101174,101539,101904,
    // 2180-2189
    102269,102635,103000,103365,103730,104096,104461,104826,105191,105557,
    // 2190-2199
    105922,106287,106652,107018,107383,107748,108113,108479,108844,109209,
    // 2200
    109574
  };
  return YearOffset[y-1900];
}

bool isLeapCpu(int y)
{
  bool YearIsLeap[] = {
    // 1900-1909
    true,false,false,false, true,false,false,false, true,false,
    // 1910-1919
    false,false, true,false,false,false, true,false,false,false,
    // 1920-1929
    true,false,false,false, true,false,false,false, true,false,
    // 1930-1939
    false,false, true,false,false,false, true,false,false,false,
    // 1940-1949
    true,false,false,false, true,false,false,false, true,false,
    // 1950-1959
    false,false, true,false,false,false, true,false,false,false,
    // 1960-1969
    true,false,false,false, true,false,false,false, true,false,
    // 1970-1979
    false,false, true,false,false,false, true,false,false,false,
    // 1980-1989
    true,false,false,false, true,false,false,false, true,false,
    // 1990-1999
    false,false, true,false,false,false, true,false,false,false,
    // 2000-2009
    true,false,false,false, true,false,false,false, true,false,
    // 2010-2019
    false,false, true,false,false,false, true,false,false,false,
    // 2020-2029
    true,false,false,false, true,false,false,false, true,false,
    // 2030-2039
    false,false, true,false,false,false, true,false,false,false,
    // 2040-2049
    true,false,false,false, true,false,false,false, true,false,
    // 2050-2059
    false,false, true,false,false,false, true,false,false,false,
    // 2060-2069
    true,false,false,false, true,false,false,false, true,false,
    // 2070-2079
    false,false, true,false,false,false, true,false,false,false,
    // 2080-2089
    true,false,false,false, true,false,false,false, true,false,
    // 2090-2099
    false,false, true,false,false,false, true,false,false,false,
    // 2100-2109
    false,false,false,false, true,false,false,false, true,false,
    // 2110-2119
    false,false, true,false,false,false, true,false,false,false,
    // 2120-2129
    true,false,false,false, true,false,false,false, true,false,
    // 2130-2139
    false,false, true,false,false,false, true,false,false,false,
    // 2140-2149
    true,false,false,false, true,false,false,false, true,false,
    // 2150-2159
    false,false, true,false,false,false, true,false,false,false,
    // 2160-2169
    true,false,false,false, true,false,false,false, true,false,
    // 2170-2179
    false,false, true,false,false,false, true,false,false,false,
    // 2180-2189
    true,false,false,false, true,false,false,false, true,false,
    // 2190-2199
    false,false, true,false,false,false, true,false,false,false,
    // 2200
    false
  };
  return YearIsLeap[y-1900];
}

bondsDateStruct intializeDateCpu(int d, int m, int y)
{
  bondsDateStruct currDate;
  currDate.day = d;
  currDate.month = m;
  currDate.year = y;
  bool leap = isLeapCpu(y);
  int offset = monthOffsetCpu(m,leap);
  currDate.dateSerialNum = d + offset + yearOffsetCpu(y);
  return currDate;
}

// ============================================================
// Benchmark driver
// ============================================================

void runBoundsEngine(const int repeat)
{
  int nBondsArray[] = {1000000};

  for (int numTime=0; numTime < 1; numTime++)
  {
    int numBonds = nBondsArray[numTime];
    printf("\nNumber of Bonds: %d\n\n", numBonds);

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (bondsYieldTermStruct*)malloc(numBonds*sizeof(bondsYieldTermStruct));
    inArgsHost.repoCurve     = (bondsYieldTermStruct*)malloc(numBonds*sizeof(bondsYieldTermStruct));
    inArgsHost.currDate      = (bondsDateStruct*)malloc(numBonds*sizeof(bondsDateStruct));
    inArgsHost.maturityDate  = (bondsDateStruct*)malloc(numBonds*sizeof(bondsDateStruct));
    inArgsHost.bondCleanPrice= (dataType*)malloc(numBonds*sizeof(dataType));
    inArgsHost.bond          = (bondStruct*)malloc(numBonds*sizeof(bondStruct));
    inArgsHost.dummyStrike   = (dataType*)malloc(numBonds*sizeof(dataType));

    srand(123);

    for (int numBond = 0; numBond < numBonds; numBond++)
    {
      dataType repoRate = 0.07;
      int repoCompounding = SIMPLE_INTEREST;
      dataType repoCompoundFreq = 1;

      bondsDateStruct bondIssueDate    = intializeDateCpu(rand() % 28 + 1, rand() % 12 + 1, 1999 - (rand() % 2));
      bondsDateStruct bondMaturityDate = intializeDateCpu(rand() % 28 + 1, rand() % 12 + 1, 2000 + (rand() % 2));
      bondsDateStruct todaysDate       = intializeDateCpu(bondMaturityDate.day-1, bondMaturityDate.month, bondMaturityDate.year);

      bondStruct bond;
      bond.startDate    = bondIssueDate;
      bond.maturityDate = bondMaturityDate;
      bond.rate         = 0.08 + ((float)rand()/(float)RAND_MAX - 0.5)*0.1;

      dataType bondCouponFrequency = 2;
      dataType bondCleanPrice      = 89.97693786;

      bondsYieldTermStruct bondCurve;
      bondCurve.refDate     = todaysDate;
      bondCurve.calDate     = todaysDate;
      bondCurve.forward     = -0.1f;
      bondCurve.compounding = COMPOUNDED_INTEREST;
      bondCurve.frequency   = bondCouponFrequency;
      bondCurve.dayCounter  = USE_EXACT_DAY;

      dataType dummyStrike = 91.5745;

      bondsYieldTermStruct repoCurve;
      repoCurve.refDate     = todaysDate;
      repoCurve.calDate     = todaysDate;
      repoCurve.forward     = repoRate;
      repoCurve.compounding = repoCompounding;
      repoCurve.frequency   = repoCompoundFreq;
      repoCurve.dayCounter  = USE_SERIAL_NUMS;

      inArgsHost.discountCurve[numBond] = bondCurve;
      inArgsHost.repoCurve[numBond]     = repoCurve;
      inArgsHost.currDate[numBond]      = todaysDate;
      inArgsHost.maturityDate[numBond]  = bondMaturityDate;
      inArgsHost.bondCleanPrice[numBond]= bondCleanPrice;
      inArgsHost.bond[numBond]          = bond;
      inArgsHost.dummyStrike[numBond]   = dummyStrike;
    }

    printf("Inputs for bond with index %d\n", numBonds/2);
    printf("Bond Issue Date: %d-%d-%d\n",
           inArgsHost.bond[numBonds/2].startDate.month,
           inArgsHost.bond[numBonds/2].startDate.day,
           inArgsHost.bond[numBonds/2].startDate.year);
    printf("Bond Maturity Date: %d-%d-%d\n",
           inArgsHost.bond[numBonds/2].maturityDate.month,
           inArgsHost.bond[numBonds/2].maturityDate.day,
           inArgsHost.bond[numBonds/2].maturityDate.year);
    printf("Bond rate: %f\n\n", inArgsHost.bond[numBonds/2].rate);

    resultsStruct resultsFromGpu;
    resultsFromGpu.dirtyPrice            = (dataType*)malloc(numBonds*sizeof(dataType));
    resultsFromGpu.accruedAmountCurrDate = (dataType*)malloc(numBonds*sizeof(dataType));
    resultsFromGpu.cleanPrice            = (dataType*)malloc(numBonds*sizeof(dataType));
    resultsFromGpu.bondForwardVal        = (dataType*)malloc(numBonds*sizeof(dataType));

    long ktimeGpu = 0;
    struct timeval start, end;

    gettimeofday(&start, NULL);
    for (int i = 0; i < repeat; i++)
      ktimeGpu += getBondsResultsGpu(inArgsHost, resultsFromGpu, numBonds);
    gettimeofday(&end, NULL);

    double timeGpu = (end.tv_sec - start.tv_sec) * 1e6 + end.tv_usec - start.tv_usec;

    printf("Average kernel execution time: %lf (ms)\n\n", ktimeGpu * 1e-3 / repeat);
    printf("Average total time (incl. transfers): %lf (ms)\n\n", timeGpu * 1e-3 / repeat);

    double totPrice = 0.0;
    for (int numBond1 = 0; numBond1 < numBonds; numBond1++)
      totPrice += resultsFromGpu.dirtyPrice[numBond1];

    printf("Sum of output dirty prices: %f\n", totPrice);
    printf("Outputs for bond with index %d:\n", numBonds/2);
    printf("Dirty Price: %f\n",       resultsFromGpu.dirtyPrice[numBonds/2]);
    printf("Accrued Amount: %f\n",    resultsFromGpu.accruedAmountCurrDate[numBonds/2]);
    printf("Clean Price: %f\n",       resultsFromGpu.cleanPrice[numBonds/2]);
    printf("Bond Forward Val: %f\n\n",resultsFromGpu.bondForwardVal[numBonds/2]);

    free(resultsFromGpu.dirtyPrice);
    free(resultsFromGpu.accruedAmountCurrDate);
    free(resultsFromGpu.cleanPrice);
    free(resultsFromGpu.bondForwardVal);
    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    runBoundsEngine(repeat);
  }
  Kokkos::finalize();
  return 0;
}
